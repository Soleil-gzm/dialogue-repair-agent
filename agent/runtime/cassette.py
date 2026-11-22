"""LLM 响应录制与回放（cassette）

为什么需要：在线评测结论要能被离线复核，但真实模型调用既花钱又不可复现。
cassette 把每次请求的响应逐行录成 JSONL，回放时按请求指纹取回同一条响应，
从而在无密钥、无网络的环境下重算出与 live 完全一致的指标与产物。

边界：
- 只记录响应，不改写请求语义；
- 回放响应标记 `cached=True`，因此不产生任何成本；
- 请求指纹只覆盖会改变模型输出的参数（messages / temperature /
  max_tokens / response_format / tools / idempotency_key），
  `stage` / `run_id` / `dialogue_key` 仅作审计字段，不参与指纹——
  否则每次运行 run_id 变化都会导致回放全部 miss。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..canonical import sha256_obj
from .llm import LLMClient, LLMResponse

MODE_RECORD = "record"
MODE_REPLAY = "replay"
MODES = (MODE_RECORD, MODE_REPLAY)

# 参与请求指纹的字段：任何一项变化都可能改变模型输出
_SIGNATURE_FIELDS = (
    "messages", "temperature", "max_tokens", "response_format",
    "tools", "idempotency_key",
)


class CassetteError(RuntimeError):
    """cassette 读写或匹配失败。"""


class CassetteMiss(CassetteError):
    """回放时找不到对应请求的响应。"""


def request_key(
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """请求指纹：只包含影响模型输出的参数。"""
    return sha256_obj({
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
        "tools": tools,
        "idempotency_key": idempotency_key,
    })


class CassetteClient:
    """按 mode 录制或回放 LLM 响应。

    - `record`：必须注入 `inner`，先真实调用，再把请求与响应追加到 cassette；
    - `replay`：不访问网络，只按请求指纹返回录制结果。
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        mode: str = MODE_REPLAY,
        inner: Optional[LLMClient] = None,
    ):
        if mode not in MODES:
            raise ValueError(f"未知 cassette mode: {mode}（可选 {MODES}）")
        if mode == MODE_RECORD and inner is None:
            raise ValueError("record 模式需要注入 inner 客户端")
        self.path = Path(path)
        self.mode = mode
        self.arm = "adhoc"
        self._inner = inner
        self._responses: Dict[str, Dict[str, Any]] = {}
        self._fh = None
        self.replayed = 0
        self.recorded = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if mode == MODE_REPLAY:
            if not self.path.is_file():
                raise CassetteMiss(f"cassette 文件不存在: {self.path}")
            self._load()
        else:
            self._fh = open(self.path, "w", encoding="utf-8")

    # ------------------------------------------------------------------ 加载
    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("key")
            if key and isinstance(rec.get("response"), dict):
                self._responses[key] = rec["response"]

    @property
    def entries(self) -> int:
        return len(self._responses)

    def set_arm(self, arm: str) -> None:
        """账本归集臂名；录制模式下透传给真实客户端。"""
        self.arm = arm
        setter = getattr(self._inner, "set_arm", None)
        if callable(setter):
            setter(arm)

    # ------------------------------------------------------------------ 调用
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout_s: float = 60.0,
        idempotency_key: Optional[str] = None,
        stage: str = "check",
        run_id: str = "adhoc",
        dialogue_key: str = "",
    ) -> LLMResponse:
        key = request_key(
            messages, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, tools=tools,
            idempotency_key=idempotency_key,
        )
        if self.mode == MODE_REPLAY:
            return self._replay(key, stage=stage)

        resp = self._inner.chat(
            messages, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, tools=tools, timeout_s=timeout_s,
            idempotency_key=idempotency_key, stage=stage,
            run_id=run_id, dialogue_key=dialogue_key,
        )
        self._append(key, messages=messages, temperature=temperature,
                     max_tokens=max_tokens, response_format=response_format,
                     tools=tools, idempotency_key=idempotency_key,
                     stage=stage, run_id=run_id, dialogue_key=dialogue_key,
                     resp=resp)
        return resp

    def _replay(self, key: str, *, stage: str) -> LLMResponse:
        rec = self._responses.get(key)
        if rec is None:
            raise CassetteMiss(
                "cassette 中找不到对应请求的响应；请确认回放使用了与录制时"
                f"相同的 prompt 与参数（stage={stage}）。"
                f"cassette={self.path}，entries={len(self._responses)}"
            )
        self.replayed += 1
        return LLMResponse(
            text=str(rec.get("text") or ""),
            model=str(rec.get("model") or "cassette"),
            input_tokens=int(rec.get("input_tokens") or 0),
            output_tokens=int(rec.get("output_tokens") or 0),
            cached=True,
            latency_ms=0,
            cached_tokens=int(rec.get("cached_tokens") or 0),
            tool_calls=list(rec.get("tool_calls") or []),
        )

    def _append(self, key: str, *, messages, temperature, max_tokens,
                response_format, tools, idempotency_key, stage, run_id,
                dialogue_key, resp: LLMResponse) -> None:
        self.recorded += 1
        record = {
            "schema": 1,
            "key": key,
            "request": {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "tools": tools,
                "idempotency_key": idempotency_key,
            },
            "response": {
                "text": resp.text,
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cached_tokens": resp.cached_tokens,
                "tool_calls": list(resp.tool_calls or []),
            },
            "meta": {
                "stage": stage,
                "run_id": run_id,
                "dialogue_key": dialogue_key,
                "arm": self.arm,
            },
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            closer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


__all__ = [
    "MODE_RECORD", "MODE_REPLAY", "MODES", "CassetteClient",
    "CassetteError", "CassetteMiss", "request_key",
]
