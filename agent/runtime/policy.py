"""决策策略

同一个循环配不同策略，是消融实验的基础：
- RulePolicy  确定性，无 LLM，作为基线（M2）
- LLMPolicy   function calling 自主决策（M3）

策略只负责"下一步做什么"，不负责执行，也不修改状态。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

ACTION_CALL_TOOL = "call_tool"
ACTION_STOP = "stop"


@dataclass(frozen=True)
class Decision:
    action: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class Policy(Protocol):
    def decide(self, state: Any) -> Decision:
        ...


class RulePolicy:
    """确定性策略：校验 → 有违规则回退原文，否则接受。

    不依赖 LLM，因此可在离线档位复现；它同时是 M3 自主决策的对照基线。
    """

    def __init__(self, check_tool: str = "check_consistency",
                 fallback_tool: str = "fallback_keep"):
        self._check = check_tool
        self._fallback = fallback_tool

    def decide(self, state: Any) -> Decision:
        called = state.called_tools()

        # 第一步：校验一致性与实体保真
        if self._check not in called:
            return Decision(
                ACTION_CALL_TOOL, self._check,
                {"before": state.original, "after": state.dialogue},
                "先校验结构不变与实体保真",
            )

        # 第二步：有违规就保留原文（宁可不变，也不能改坏）
        if state.issues and self._fallback not in called:
            return Decision(
                ACTION_CALL_TOOL, self._fallback,
                {"original": state.original},
                f"检测到 {len(state.issues)} 处违规，回退原文",
            )

        return Decision(ACTION_STOP, reason="校验通过，接受当前版本")
