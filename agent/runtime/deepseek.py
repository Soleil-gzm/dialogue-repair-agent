"""DeepSeek 客户端

职责边界：
- 只做一次 HTTP 调用 + 重试 + 计量，不含 prompt 设计，也不做业务判断
- 每次调用（成功/失败/缓存命中）都写一行 Token 账本
- 密钥只从环境变量读，且永不进日志

重试策略：只重试"可恢复"错误（网络异常、429、5xx）。
4xx 参数错误重试没有意义，直接失败——避免把配置问题伪装成抖动。
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from .config import LLMSettings
from .ledger import LedgerEntry, Pricing, TokenLedger
from .llm import LLMResponse
from .pricing import select_pricing

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """LLM 调用失败基类。"""


class LLMHTTPError(LLMError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


class LLMParseError(LLMError):
    """响应结构不符合预期。"""


class LLMConfigError(LLMError):
    """请求参数不合法，重试无意义，应在本地提前拦截。"""


def _sha256(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, LLMHTTPError):
        return exc.status in RETRYABLE_STATUS
    return False


@dataclass
class DeepSeekClient:
    settings: LLMSettings = field(default_factory=LLMSettings)
    api_key: Optional[str] = None
    ledger: Optional[TokenLedger] = None
    # 不传则按调用时刻自动选高峰/空闲价
    pricing: Optional[Pricing] = None

    # 测试注入点
    transport: Optional[httpx.BaseTransport] = None
    max_attempts: int = 3
    backoff_base: float = 0.5

    # 进程内缓存：同一请求不重复付费
    cache: Dict[str, LLMResponse] = field(default_factory=dict)
    _seq: int = 0

    # ------------------------------------------------------------------ 初始化
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = self.settings.api_key()

    def _active_pricing(self) -> Pricing:
        """显式注入的价目优先；否则按每次调用时刻取档，跨时段也不会串价。"""
        return self.pricing or select_pricing()

    # ------------------------------------------------------------------ 调用
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout_s: Optional[float] = None,
        idempotency_key: Optional[str] = None,
        stage: str = "check",
        run_id: str = "adhoc",
        dialogue_key: str = "",
    ) -> LLMResponse:
        payload = self._build_payload(
            messages, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, tools=tools,
        )
        req_hash = _sha256({"payload": payload, "key": idempotency_key})

        if req_hash in self.cache:
            hit = self.cache[req_hash]
            self._write_ledger(
                stage=stage, run_id=run_id, dialogue_key=dialogue_key,
                input_tokens=hit.input_tokens, output_tokens=hit.output_tokens,
                cached_tokens=hit.cached_tokens,
                latency_ms=0, retries=0, http_status=200, cached=True,
                request_hash=req_hash,
            )
            return LLMResponse(
                text=hit.text, model=hit.model,
                input_tokens=hit.input_tokens, output_tokens=hit.output_tokens,
                cached=True, latency_ms=0, cached_tokens=hit.cached_tokens,
            )

        attempts = {"n": 0}

        def _before_sleep(_retry_state):
            attempts["n"] += 1

        retryer = Retrying(
            stop=stop_after_attempt(max(1, self.max_attempts)),
            wait=wait_exponential(multiplier=self.backoff_base, max=8),
            retry=retry_if_exception(_is_retryable),
            before_sleep=_before_sleep,
            reraise=True,
        )

        t0 = time.monotonic()
        try:
            resp = retryer(self._post, payload, timeout_s)
        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            self._write_ledger(
                stage=stage, run_id=run_id, dialogue_key=dialogue_key,
                input_tokens=0, output_tokens=0, cached_tokens=0,
                latency_ms=latency_ms, retries=attempts["n"],
                http_status=getattr(e, "status", None), cached=False,
                request_hash=req_hash, ok=False, error_type=type(e).__name__,
            )
            raise

        latency_ms = int((time.monotonic() - t0) * 1000)
        result = self._parse(resp)
        self.cache[req_hash] = result
        self._write_ledger(
            stage=stage, run_id=run_id, dialogue_key=dialogue_key,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            latency_ms=latency_ms, retries=attempts["n"],
            http_status=resp.status_code, cached=False, request_hash=req_hash,
        )
        return LLMResponse(
            text=result.text, model=result.model,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            cached=False, latency_ms=latency_ms, cached_tokens=result.cached_tokens,
        )

    # ------------------------------------------------------------------ 内部
    def _build_payload(self, messages, *, temperature, max_tokens,
                       response_format, tools) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if self.settings.json_mode:
            fmt = response_format or {"type": "json_object"}
            # DeepSeek 的硬约束：json_object 要求 prompt 里出现 "json" 字样，
            # 否则服务端直接 400。本地先拦，避免把配置问题伪装成网络抖动。
            if fmt.get("type") == "json_object" and not self._mentions_json(messages):
                raise LLMConfigError(
                    "json_mode 要求 prompt 中包含 'json' 字样（DeepSeek 接口约束），"
                    "请修改 prompt 或显式传入 response_format=None 并关闭 json_mode"
                )
            payload["response_format"] = fmt
        elif response_format is not None:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        return payload

    @staticmethod
    def _mentions_json(messages: List[Dict[str, str]]) -> bool:
        return any("json" in str(m.get("content", "")).lower() for m in messages)

    def _post(self, payload: Dict[str, Any], timeout_s: Optional[float]) -> httpx.Response:
        timeout = self.settings.timeout_s if timeout_s is None else timeout_s
        with httpx.Client(
            base_url=self.settings.base_url.rstrip("/"),
            timeout=timeout,
            transport=self.transport,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            resp = client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            raise LLMHTTPError(resp.status_code, resp.text)
        return resp

    def _parse(self, resp: httpx.Response) -> LLMResponse:
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMParseError(f"响应结构异常: {e}") from e
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text or "",
            model=data.get("model") or self.settings.model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=self._cache_hit_tokens(usage),
        )

    @staticmethod
    def _cache_hit_tokens(usage: Dict[str, Any]) -> int:
        """DeepSeek 用 prompt_cache_hit_tokens；兼容 OpenAI 风格的 details.cached_tokens。"""
        hit = usage.get("prompt_cache_hit_tokens")
        if hit is None:
            details = usage.get("prompt_tokens_details") or {}
            hit = details.get("cached_tokens")
        try:
            return int(hit or 0)
        except (TypeError, ValueError):
            return 0

    def _write_ledger(self, *, stage, run_id, dialogue_key, input_tokens,
                      output_tokens, cached_tokens, latency_ms, retries,
                      http_status, cached, request_hash, ok=True,
                      error_type=None) -> None:
        if self.ledger is None:
            return
        pricing = self._active_pricing()
        self._seq += 1
        self.ledger.record(LedgerEntry(
            run_id=run_id,
            dialogue_key=dialogue_key,
            seq=self._seq,
            stage=stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            model=self.settings.model,
            provider=self.settings.provider,
            latency_ms=latency_ms,
            retries=retries,
            http_status=http_status,
            cached=cached,
            cost_cny=pricing.cost(input_tokens, output_tokens, cached_tokens),
            price_table_version=pricing.version,
            request_hash=request_hash,
            ok=ok,
            error_type=error_type,
        ))
