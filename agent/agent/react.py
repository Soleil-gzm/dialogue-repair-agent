"""显式 ReAct / Planning&Executor 架构化。

ReActAgent 把现有 AgentRuntime 包装成显式的 ReAct 变体（observe→think→act→observe），
PlanningExecutorAgent 则是两阶段：先规划候选轮，再只对候选轮做精细检查。

设计约束：
  - 不硬编码任何轮次。业务侧的特殊轮（如私有轨随机开场句）必须由调用方
    通过 exclude_turns 传入，而不是写死在架构层。
  - 规划必须真正改变执行行为，否则只是装饰：
    plan 产出的候选集会换算成"判官忽略集"，让判官只检查候选轮。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..runtime.loop import AgentResult, AgentRuntime


@dataclass
class ReActStep:
    """一次 ReAct step 的完整记录（Thought / Action / Observation）。"""
    step: int
    thought: str
    action: str
    args: Dict[str, Any]
    observation: Dict[str, Any]
    tokens: int = 0


class ReActAgent:
    """显式 ReAct 变体：包装 AgentRuntime 的 think→act→observe 循环。

    Thought = policy.decide 的 reason；Action = 工具名；Observation = 工具结果。
    循环的三道护栏（预算 / 重复调用 / trace）由 AgentRuntime 提供。
    """

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def run(self, dialogue: Dict[str, Any],
            reference: Optional[Dict[str, Any]] = None) -> AgentResult:
        return self.runtime.run(dialogue, reference=reference)

    @staticmethod
    def extract_steps(trace: List[Dict[str, Any]]) -> List[ReActStep]:
        """从 runtime.trace 解析 ReAct step 序列，便于调试与展示。"""
        steps: List[ReActStep] = []
        pending: Optional[Dict[str, Any]] = None
        idx = 0
        for ev in trace:
            event = ev.get("event")
            if event == "decide" and ev.get("action") == "call_tool":
                pending = ev
            elif event == "tool_result" and pending is not None:
                idx += 1
                steps.append(ReActStep(
                    step=idx,
                    thought=str(pending.get("reason", "")),
                    action=str(pending.get("tool", "")),
                    args=pending.get("args", {}) or {},
                    observation={
                        "ok": ev.get("ok"),
                        "terminal": ev.get("terminal"),
                        "latency_ms": ev.get("latency_ms"),
                    },
                    tokens=int(ev.get("tokens", 0) or 0),
                ))
                pending = None
        return steps


# =============================================================================
# Planning & Executor
# =============================================================================

@dataclass
class Plan:
    candidates: List[int] = field(default_factory=list)
    rationale: str = ""


class PlanningExecutorAgent:
    """两阶段 Agent：Plan（选候选轮）→ Execute（只对候选轮做精细检查）。

    runtime_factory: Callable[[frozenset], AgentRuntime]
        接收"判官应忽略的轮次集合"，返回装配好的 runtime。
        这样 Planning 不必依赖 benchmark 模块，也不会与 build_runtime 循环导入。

    exclude_turns: 调用方显式声明的排除轮（如业务伪影轮），默认空。
    """

    def __init__(self, runtime_factory: Callable[[frozenset], AgentRuntime],
                 *, exclude_turns: Any = ()):
        self.runtime_factory = runtime_factory
        self.exclude_turns = tuple(sorted({int(t) for t in exclude_turns}))

    def plan(self, dialogue: Dict[str, Any]) -> Plan:
        msgs = dialogue.get("messages") or []
        excluded = set(self.exclude_turns)
        candidates = [
            i for i, m in enumerate(msgs)
            if isinstance(m, dict) and m.get("role") == "assistant" and i not in excluded
        ]
        return Plan(candidates=candidates,
                    rationale=f"候选 assistant 轮（排除 {sorted(excluded)}）")

    def execute(self, dialogue: Dict[str, Any], plan: Plan,
                reference: Optional[Dict[str, Any]] = None) -> AgentResult:
        msgs = dialogue.get("messages") or []
        candidates = set(plan.candidates)
        all_assistant = [
            i for i, m in enumerate(msgs)
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        # 非候选轮交给判官忽略——规划由此真正收窄了检查范围
        ignored = frozenset(t for t in all_assistant if t not in candidates)
        runtime = self.runtime_factory(ignored)
        return runtime.run(dialogue, reference=reference)

    def run(self, dialogue: Dict[str, Any],
            reference: Optional[Dict[str, Any]] = None) -> AgentResult:
        return self.execute(dialogue, self.plan(dialogue), reference=reference)


def build_agent(arm: str, runtime: AgentRuntime,
                *, planning: bool = False) -> ReActAgent | PlanningExecutorAgent:
    """按需选择架构；planning=True 时需改用 runtime_factory 形式的构造方式。"""
    if planning:
        raise ValueError(
            "PlanningExecutorAgent 需要 runtime_factory，请直接实例化并传入 "
            "lambda ignored: build_runtime(...) "
        )
    return ReActAgent(runtime)
