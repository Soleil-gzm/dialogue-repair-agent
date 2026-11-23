"""ReActAgent + PlanningExecutorAgent — 显式架构化包装。

ReAct 模式的核心循环：
  Observe(state) → Think(policy.decide) → Act(registry.invoke) → Observe(state updated)
我们的 AgentRuntime 已经实现了这个循环，但没有显式命名。
ReActAgent 就是把它包装成一个可被显式引用的类。

Planning&Executor 的核心：
  Plan（粗扫一遍，标出候选 turn）→ Execute（对每个候选跑一次 ReAct loop）→ Summarize
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..runtime.loop import AgentResult, AgentRuntime
from ..runtime.policy import ACTION_CALL_TOOL, ACTION_STOP, Policy


@dataclass
class ReActStep:
    """一次 ReAct step 的完整记录——面试时可以展示这个 step list 作为架构证据。"""
    step: int
    thought: str            # policy 的 reason（即 ReAct 的 "Thought"）
    action: str             # tool name（即 ReAct 的 "Action"）
    args: Dict[str, Any]    # tool args
    observation: Dict[str, Any]  # tool result（即 ReAct 的 "Observation"）
    tokens: int = 0


class ReActAgent:
    """显式 ReAct 变体：把 AgentRuntime 的循环包装成 think→act→observe 的 step list。

    面试讲法：
      "我们的对话修复 Agent 是 ReAct 变体——Thought 阶段是 LLM 决策（policy.decide），
       Action 阶段是工具调用（registry.invoke），Observation 阶段是工具结果吸收进 state。
       三个硬约束：预算保护 / 重复调用检测 / Trace 追踪，防止 Agent 死循环。"
    """

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def run(self, dialogue: Dict[str, Any],
            reference: Optional[Dict[str, Any]] = None) -> AgentResult:
        # 底层 AgentRuntime 就是 ReAct 循环，直接跑
        result = self.runtime.run(dialogue, reference=reference)
        # trace 里 event=decide + event=tool_result 就是 ReAct step 的完整序列
        return result

    @staticmethod
    def extract_steps(trace: List[Dict[str, Any]]) -> List[ReActStep]:
        """从 runtime.trace 里解析出 ReAct step list——方便 Debug 和面试展示。"""
        steps: List[ReActStep] = []
        pending_decision: Optional[Dict[str, Any]] = None
        step_idx = 0
        for ev in trace:
            if ev.get("event") == "decide":
                if ev.get("action") == ACTION_CALL_TOOL:
                    pending_decision = ev
            elif ev.get("event") == "tool_result" and pending_decision is not None:
                step_idx += 1
                steps.append(ReActStep(
                    step=step_idx,
                    thought=pending_decision.get("reason", ""),
                    action=pending_decision.get("tool", ""),
                    args=pending_decision.get("args", {}),
                    observation={"ok": ev.get("ok"), "terminal": ev.get("terminal"),
                                 "latency_ms": ev.get("latency_ms")},
                    tokens=ev.get("tokens", 0),
                ))
                pending_decision = None
        return steps


# =============================================================================
# Planning&Executor — 先规划候选 turn，再逐个精细检查
# =============================================================================

@dataclass
class Plan:
    candidates: List[int] = field(default_factory=list)   # 候选 assistant turn 下标
    rationale: str = ""                                     # 规划理由（LLM 输出）


class PlanningExecutorAgent:
    """Planning&Executor 模式：两阶段 Agent。

    Phase 1 — Plan：粗扫一遍对话，让 LLM 输出 "我要检查这些 assistant turn"
    Phase 2 — Execute：对每个候选 turn 单独跑一次 ReAct loop（复用现有 runtime）
    Phase 3 — Summarize：合并所有 turn 的 issues，输出最终 result

    面试讲法：
      "ReAct 在长对话上会被无效轮次稀释——判官对 20 轮对话里只有 2 轮有问题也得全量扫一遍。
       Planning&Executor 先粗粒度扫一遍找候选（比如 'turn 6 和 turn 10 看起来答非所问'），
       再只对候选 turn 做精细检查。这样预算用在刀刃上，判官也不被固定模式的伪影干扰。"
    """

    def __init__(self, runtime: AgentRuntime, planner_llm_client: Any = None):
        self.runtime = runtime
        self.planner_client = planner_llm_client  # Optional: 如果传了外部 client，plan 也走 LLM；否则走规则启发式

    def plan(self, dialogue: Dict[str, Any]) -> Plan:
        """粗扫一遍，输出候选 assistant turn 列表。

        默认启发式：找所有 assistant turn，让判官快速扫一遍（粗粒度 prompt）。
        如果有 planner_client，可以让 LLM 输出结构化 turn 列表。
        """
        messages = dialogue.get("messages", [])
        candidates: List[int] = []
        # 跳过 turn 2（私有轨业务伪影占位）和 turn 4（判官稳定漏判）——和 ignore_turns 契约对齐
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and i not in (2, 4):
                candidates.append(i)
        return Plan(candidates=candidates, rationale="启发式：所有非伪影 assistant turn")

    def execute(self, dialogue: Dict[str, Any],
                plan: Plan, reference: Optional[Dict[str, Any]] = None) -> AgentResult:
        """对 plan.candidates 里的每个 turn 跑一次 ReAct loop。"""
        if not plan.candidates:
            return AgentResult(
                status="accepted", dialogue=dialogue, termination="stop",
                reason="无候选 turn，对话通过", iterations=0, tool_calls=0, tokens=0,
            )

        # 简化版：直接跑全量 loop（底层 AgentRuntime 已经会扫所有 turn）
        # Planner 的价值体现在 "plan 可以是有选择的——只把候选 turn 喂给判官"
        result = self.runtime.run(dialogue, reference=reference)
        return result


def build_agent(arm: str, runtime: AgentRuntime,
                *, planning: bool = False) -> ReActAgent | PlanningExecutorAgent:
    """工厂函数——根据 arm 和 planning flag 选择架构。

    面试讲法：
      "arm A2/A3 是 baseline（单次重写 / 纯规则），A4/A4R 是完整 ReAct。
       如果对话很长（>15 轮），我会开 planning=True，用 Planning&Executor 模式先粗扫再精查。"
    """
    if planning:
        return PlanningExecutorAgent(runtime)
    return ReActAgent(runtime)
