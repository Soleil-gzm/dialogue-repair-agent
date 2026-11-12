"""LLM 客户端抽象

三种实现，对应三种运行档位：
- FakeLLMClient    ：单测 / CI / 离线复现，确定性
- DeepSeekClient   ：M3 接入，真实调用（本期不实现）
- CassetteClient   ：回放真实响应（M3 之后）

协议先行：上层只依赖 LLMClient，不关心具体厂商。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class LLMResponse:
    """一次调用的结果。cached=True 时成本应记 0。"""

    text: str
    model: str = "fake"
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(Protocol):
    """所有客户端必须实现的最小接口。"""

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
    ) -> LLMResponse:
        ...


class FakeLLMClient:
    """确定性假客户端。

    用途：
    - 让工具与循环逻辑可以脱离网络单测
    - 让 Benchmark 在无密钥环境下可复现

    匹配规则：先按 idempotency_key，再按 messages 内容哈希，最后用 default。
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default: str = "{}",
        model: str = "fake",
    ):
        self._responses = dict(responses or {})
        self._default = default
        self._model = model
        self.calls: List[Dict[str, Any]] = []

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
    ) -> LLMResponse:
        self.calls.append({
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "tools": tools,
            "idempotency_key": idempotency_key,
        })

        text = None
        if idempotency_key is not None:
            text = self._responses.get(idempotency_key)
        if text is None:
            text = self._responses.get(self._hash(messages), self._default)

        # 粗略估算 token：字符数/4，只为让计量链路可测
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=max(1, len(text) // 4),
        )

    @staticmethod
    def _hash(messages: List[Dict[str, str]]) -> str:
        import hashlib
        import json

        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def call_count(self) -> int:
        return len(self.calls)
