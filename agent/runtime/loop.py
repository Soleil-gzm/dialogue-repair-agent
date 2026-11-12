"""Agent 决策循环

观察 → 决策 → 调用 → 反馈，直到显式终止或触顶。

三条硬约束：
1. 每次迭代先查预算，触顶即停（不允许"再试一次"）
2. 同一 (工具, 参数指纹) 连续重复即判死循环并终止
3. 每一步都写 Trace，包括被拒的决策与异常
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .budget import Budget
from .policy import ACTION_CALL_TOOL, ACTION_STOP, Policy
from .state import AgentState
from .trace import TraceWriter

# 结果状态
ACCEPTED = "accepted"
FALLBACK = "fallback"
STOPPED = "stopped"
FAILED = "failed"

# 终止原因
TERM_STOP = "stop"
TERM_LOOP_DETECTED = "loop_detected"
TERM_TOOL_TERMINAL = "tool_terminal"
TERM_ERROR = "error"

MAX_REPEAT_SAME_CALL = 2


@dataclass
class AgentResult:
    status: str
    dialogue: Dict[str, Any]
    termination: str
    reason: str = ""
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (ACCEPTED, FALLBACK)


def _args_fingerprint(tool: str, args: Dict[str, Any]) -> str:
    """参数指纹：用于识别"同一决策反复出现"。"""
    payload = json.dumps({"tool": tool, "args": args}, ensure_ascii=False, sort_keys=True)
    import hashlib
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class AgentRuntime:
    def __init__(
        self,
        policy: Policy,
        *,
        budget: Optional[Budget] = None,
        trace: Optional[TraceWriter] = None,
        registry: Any = None,
    ):
        from .registry import ToolRegistry  # 延迟导入，便于测试替换

        self.policy = policy
        self.budget_template = budget or Budget()
        self.trace = trace or TraceWriter()
        self.registry = registry or ToolRegistry

    def run(self, dialogue: Dict[str, Any],
            reference: Optional[Dict[str, Any]] = None) -> AgentResult:
        """执行一次决策循环。

        Args:
            dialogue: 待校验/修复的候选对话
            reference: 原始参照（校验基准与回退目标）。缺省时以 dialogue 自身为基准，
                       此时一致性校验必然通过，只用于跑通链路。
        """
        import copy

        state = AgentState(
            dialogue=copy.deepcopy(dialogue),
            original=copy.deepcopy(reference if reference is not None else dialogue),
            budget=Budget(
                max_iterations=self.budget_template.max_iterations,
                max_tool_calls=self.budget_template.max_tool_calls,
                max_tokens=self.budget_template.max_tokens,
                max_wall_clock_s=self.budget_template.max_wall_clock_s,
            ),
        )
        self.trace.write({"event": "start", "key": state.key})

        termination, reason = TERM_STOP, ""
        repeat_seen: Dict[str, int] = {}

        while True:
            exhausted = state.budget.exhausted()
            if exhausted:
                termination, reason = exhausted, "触达预算上限"
                self.trace.write({"event": "budget_exhausted", "reason": exhausted,
                                  "key": state.key})
                break

            try:
                decision = self.policy.decide(state)
            except Exception as e:  # 策略异常不得静默
                termination, reason = TERM_ERROR, f"策略异常: {e}"
                self.trace.write({"event": "policy_error", "error": str(e), "key": state.key})
                break

            self.trace.write({
                "event": "decide",
                "key": state.key,
                "action": decision.action,
                "tool": decision.tool,
                "reason": decision.reason,
            })

            if decision.action == ACTION_STOP:
                termination, reason = TERM_STOP, decision.reason
                break

            if decision.action != ACTION_CALL_TOOL:
                termination, reason = TERM_ERROR, f"未知动作: {decision.action}"
                break

            sig = _args_fingerprint(decision.tool, decision.args)
            repeat_seen[sig] = repeat_seen.get(sig, 0) + 1
            if repeat_seen[sig] > MAX_REPEAT_SAME_CALL:
                termination = TERM_LOOP_DETECTED
                reason = f"同一决策重复 {repeat_seen[sig]} 次: {decision.tool}"
                self.trace.write({"event": "loop_detected", "tool": decision.tool,
                                  "key": state.key})
                break

            state.budget.tick_iteration()
            state.budget.consume(tool_calls=1)

            t0 = time.monotonic()
            try:
                result = self.registry.invoke(decision.tool, state, decision.args)
            except Exception as e:
                termination, reason = TERM_ERROR, f"工具异常: {e}"
                self.trace.write({"event": "tool_error", "tool": decision.tool,
                                  "error": str(e), "key": state.key})
                state.budget.consume(tokens=0)
                break

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            state.budget.consume(tokens=int(result.get("tokens", 0) or 0))
            self._absorb(state, decision.tool, decision.args, result)

            self.trace.write({
                "event": "tool_result",
                "key": state.key,
                "tool": decision.tool,
                "ok": bool(result.get("ok", True)),
                "terminal": bool(result.get("terminal", False)),
                "latency_ms": elapsed_ms,
                "tokens": int(result.get("tokens", 0) or 0),
            })

            if result.get("terminal"):
                termination = TERM_TOOL_TERMINAL
                reason = f"工具 {decision.tool} 终止了流程"
                break

        status = self._final_status(termination, state)
        self.trace.write({"event": "end", "key": state.key,
                          "status": status, "termination": termination})

        return AgentResult(
            status=status,
            dialogue=state.result_dialogue or state.dialogue,
            termination=termination,
            reason=reason,
            iterations=state.budget.iterations,
            tool_calls=state.budget.tool_calls,
            tokens=state.budget.tokens,
            issues=list(state.issues),
            trace=self.trace.records,
        )

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _absorb(state: AgentState, tool: str, args: Dict[str, Any],
                result: Dict[str, Any]) -> None:
        """把工具结果吸收进状态：抽出一致性违规，接收回退结果。"""
        if not result.get("ok", True):
            for v in result.get("violations", []) or []:
                state.issues.append(v)
        if result.get("status") == FALLBACK and isinstance(result.get("dialogue"), dict):
            state.result_dialogue = result["dialogue"]
            state.status = FALLBACK
        state.record(tool, args, result)

    @staticmethod
    def _final_status(termination: str, state: AgentState) -> str:
        if state.status == FALLBACK:
            return FALLBACK
        if termination in (TERM_ERROR,):
            return FAILED
        if termination == TERM_STOP:
            return ACCEPTED
        return STOPPED
