"""决策策略

同一个循环配不同策略，是消融实验的基础：
- RulePolicy  确定性，无 LLM，作为基线（M2）
- LLMPolicy   function calling 自主决策（M3）

策略只负责"下一步做什么"，不负责执行，也不修改状态。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence

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
    """确定性策略：依次校验 → 有违规则回退原文，否则接受。

    不依赖 LLM，因此可在离线档位复现；它同时是 M3 自主决策的对照基线。
    校验链默认"先保真、后连贯"：保真校验是硬约束，任何重写都必须先过。
    """

    def __init__(self,
                 checks: Sequence[str] = ("check_consistency", "check_coherence"),
                 fallback_tool: str = "fallback_keep"):
        self._checks = tuple(checks)
        self._fallback = fallback_tool

    def decide(self, state: Any) -> Decision:
        called = state.called_tools()

        # 第一步：按链顺序把校验跑完
        for tool in self._checks:
            if tool not in called:
                return Decision(
                    ACTION_CALL_TOOL, tool,
                    {"before": state.original, "after": state.dialogue}
                    if tool == "check_consistency"
                    else {"dialogue": state.dialogue},
                    "先校验结构不变与实体保真" if tool == "check_consistency"
                    else "再校验对话自身连贯性",
                )

        # 第二步：有违规就保留原文（宁可不变，也不能改坏）
        if state.issues and self._fallback not in called:
            return Decision(
                ACTION_CALL_TOOL, self._fallback,
                {"original": state.original},
                f"检测到 {len(state.issues)} 处违规，回退原文",
            )

        return Decision(ACTION_STOP, reason="校验通过，接受当前版本")


class LLMPolicyError(RuntimeError):
    """模型决策不符合契约（非工具、未知工具、既不调用也不结束）。"""


SYSTEM_PROMPT = (
    "你是对话数据修复 Agent 的决策器。每一步只能做一件事：调用一个工具，或结束流程。\n"
    "规则：\n"
    "1. 先做保真校验，再做连贯性校验；两个校验都完成前不要回退。\n"
    "2. 任一校验报出违规，就调用 fallback_keep 保留原文，不要尝试改写。\n"
    "3. 没有任何违规时结束流程（不要调用工具，直接给出简短理由）。\n"
    "4. 工具入参由运行时注入，你只需要选择工具名，不要自己拼参数。\n"
    "5. 每个工具最多调用一次。"
)

MAX_ISSUES_IN_PROMPT = 8
MAX_REASON_CHARS = 200


class LLMPolicy:
    """function calling 决策策略：让模型选工具，与 RulePolicy 构成消融对照。

    设计取舍：**模型只选工具名，入参由运行时注入**。
    原因是让模型复述整段对话既贵又容易改坏内容（把"重构参数"当成"改写数据"），
    而这里要测的是"决策能力"，不是"抄写能力"。
    """

    def __init__(self, client: Any, *,
                 tools: Optional[Sequence[str]] = None,
                 max_tokens: int = 1024,
                 run_id: str = "adhoc"):
        self.client = client
        self.tools = tuple(tools) if tools else None
        self.max_tokens = max_tokens
        self.run_id = run_id

    def decide(self, state: Any) -> Decision:
        resp = self.client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": self.observe(state)}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            tools=self._tool_specs(),
            stage="decide",
            run_id=self.run_id,
            dialogue_key=state.key,
        )

        if resp.tool_calls:
            call = resp.tool_calls[0]
            name = call.get("name")
            if name not in self._tool_names():
                raise LLMPolicyError(f"模型选择了未注册的工具: {name!r}")
            if name in state.called_tools():
                raise LLMPolicyError(f"模型重复调用工具: {name}")
            return Decision(
                ACTION_CALL_TOOL, name,
                self._build_args(name, state),
                (resp.text or "").strip()[:MAX_REASON_CHARS] or f"LLM 选择 {name}",
            )

        reason = (resp.text or "").strip()
        if not reason:
            raise LLMPolicyError("模型既没有调用工具，也没有给出结束理由")
        return Decision(ACTION_STOP, reason=reason[:MAX_REASON_CHARS])

    # ------------------------------------------------------------------ 内部
    def _tool_names(self) -> set:
        if self.tools:
            return set(self.tools)
        from .registry import ToolRegistry
        return set(ToolRegistry.list_tools())

    def _tool_specs(self) -> Sequence[dict]:
        """只暴露工具名与描述，参数留空：入参由运行时注入。"""
        from .registry import ToolRegistry

        allowed = self._tool_names()
        specs = []
        for spec in ToolRegistry.specs():
            fn = spec["function"]
            if fn["name"] not in allowed:
                continue
            specs.append({
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn["description"],
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        return specs

    @staticmethod
    def _build_args(tool: str, state: Any) -> Dict[str, Any]:
        if tool == "check_consistency":
            return {"before": state.original, "after": state.dialogue}
        if tool == "check_coherence":
            return {"dialogue": state.dialogue}
        if tool == "fallback_keep":
            return {"original": state.original}
        raise LLMPolicyError(f"没有为工具 {tool} 定义入参构造规则")

    @staticmethod
    def observe(state: Any) -> str:
        """把状态压成紧凑观测：只给决策需要的信息，不重复整段对话。"""
        msgs = list((state.dialogue or {}).get("messages") or [])
        roles = "/".join(str(m.get("role")) for m in msgs)
        lines = [
            f"对话标识: {state.key}",
            f"消息数: {len(msgs)}，角色序列: {roles}",
        ]

        if state.history:
            done = "、".join(
                f"{h.get('tool')}({'ok' if h.get('ok') else '有违规'})"
                for h in state.history
            )
            lines.append(f"已执行工具: {done}")
        else:
            lines.append("已执行工具: 无")

        if state.issues:
            lines.append(f"已发现违规 {len(state.issues)} 处：")
            for v in state.issues[:MAX_ISSUES_IN_PROMPT]:
                lines.append(f"- {v.get('rule')} @turn {v.get('turn')}: {v.get('detail')}")
        else:
            lines.append("已发现违规: 无")

        return "\n".join(lines)
