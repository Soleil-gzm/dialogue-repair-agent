"""agent.agent — 显式 ReAct / Planning&Executor 架构化。

当前 runtime 里的 AgentRuntime 本质上就是一个 ReAct loop：
  observe (state) → think (policy.decide) → act (tool call) → observe (state 更新)
但没有在类名/命名空间里显式体现 ReAct 模式，也没有 Planning 层。

本包做两件事：
  1. 把现有 AgentRuntime 包装成显式的 ReActAgent 子类——面试时能讲 "我实现了 ReAct 的哪个变体"
  2. 加 Planner 子类——先让 judge 粗扫一遍标出候选 turn 列表，再对候选 turn 做精细检查

面试时能讲清：
  - 我们的 ReAct：observe → think (LLM policy) → act (registry.invoke) → update state，loop 3 个硬约束（预算/重复/trace）
  - 我们的 Planning&Executor：plan 阶段先粗粒度扫一遍 → 候选 turn 列表 → executor 对每个候选做精细检查
  - planner 的 plan 本身也是一个 LLM 调用（让模型输出 "我要检查这些 turn"），executor 用现有 loop 对每个 turn 跑一次
"""

from .react import ReActAgent, PlanningExecutorAgent, build_agent

__all__ = ["ReActAgent", "PlanningExecutorAgent", "build_agent"]
