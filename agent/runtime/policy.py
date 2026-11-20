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
                 fallback_tool: str = "fallback_keep",
                 *,
                 rewrite_tool: Optional[str] = None,
                 max_rewrites_per_turn: int = 2,
                 max_rewrite_rounds: int = 4):
        self._checks = tuple(checks)
        self._fallback = fallback_tool
        # 不传重写工具时保持旧行为：只检查 + 回退，完全确定性
        self._rewrite = rewrite_tool
        self._max_per_turn = max(1, int(max_rewrites_per_turn))
        self._max_rounds = max(1, int(max_rewrite_rounds))

    def decide(self, state: Any) -> Decision:
        if self._rewrite is None:
            return self._decide_simple(state)
        return self._decide_rewrite(state)

    # ------------------------------------------------------------------ 旧行为
    def _decide_simple(self, state: Any) -> Decision:
        called = state.called_tools()

        # 第一步：按链顺序把校验跑完
        for tool in self._checks:
            if tool not in called:
                return self._check_decision(tool, state)

        # 第二步：有违规就保留原文（宁可不变，也不能改坏）
        if state.issues and self._fallback not in called:
            return self._fallback_decision(state)

        return Decision(ACTION_STOP, reason="校验通过，接受当前版本")

    # ------------------------------------------------------------------ 多轮修复
    def _decide_rewrite(self, state: Any) -> Decision:
        history = list(state.history or [])
        names = [h.get("tool") for h in history]

        if self._fallback in names:
            return Decision(ACTION_STOP, reason="已回退原文，结束流程")

        last_rewrite = max(
            (i for i, name in enumerate(names) if name == self._rewrite),
            default=-1,
        )
        # 重写失败说明模型给不出可用结果，直接回退，不再空转复检
        if last_rewrite >= 0 and not history[last_rewrite].get("ok", True):
            return self._fallback_decision(
                state, reason="重写失败，回退原文",
            )

        # 本轮 = 最近一次重写之后；rewrite 成功会清空 issues，必须重跑校验
        done = {name for name in names[last_rewrite + 1:]}
        for tool in self._checks:
            if tool not in done:
                return self._check_decision(tool, state)

        if not state.issues:
            return Decision(ACTION_STOP, reason="校验通过，接受当前版本")

        turn = self._target_turn(state)
        if turn is None:
            return self._fallback_decision(
                state, reason="违规无法定位到 assistant 轮次，回退原文",
            )

        rewrites = [h for h in history if h.get("tool") == self._rewrite]
        done_rewrites = [h for h in rewrites if h.get("ok", True)]
        same_turn = sum(1 for h in done_rewrites if h.get("turn") == turn)
        if same_turn >= self._max_per_turn:
            return self._fallback_decision(
                state, reason="同一轮重写已达上限，回退原文",
            )
        if len(done_rewrites) >= self._max_rounds:
            return self._fallback_decision(
                state, reason="重写轮次已达上限，回退原文",
            )

        return Decision(
            ACTION_CALL_TOOL, self._rewrite,
            {"dialogue": state.dialogue, "turn": turn, "issues": list(state.issues)},
            f"第 {turn} 轮存在 {len(state.issues)} 处违规，重写该轮",
        )

    # ------------------------------------------------------------------ 辅助
    def _check_decision(self, tool: str, state: Any) -> Decision:
        return Decision(
            ACTION_CALL_TOOL, tool,
            {"before": state.original, "after": state.dialogue}
            if tool == "check_consistency"
            else {"dialogue": state.dialogue},
            "先校验结构不变与实体保真" if tool == "check_consistency"
            else "再校验对话自身连贯性",
        )

    def _fallback_decision(self, state: Any, reason: str = "") -> Decision:
        return Decision(
            ACTION_CALL_TOOL, self._fallback,
            {"original": state.original},
            reason or f"检测到 {len(state.issues)} 处违规，回退原文",
        )

    @staticmethod
    def _target_turn(state: Any) -> Optional[int]:
        msgs = list((state.dialogue or {}).get("messages") or [])
        for issue in state.issues:
            turn = (issue or {}).get("turn")
            if isinstance(turn, bool) or not isinstance(turn, int):
                continue
            if 0 <= turn < len(msgs) and isinstance(msgs[turn], dict) \
                    and msgs[turn].get("role") == "assistant":
                return turn
        return None


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

# 允许改写时的决策规则：能修就修，修不动才回退
REWRITE_SYSTEM_PROMPT = (
    "你是对话数据修复 Agent 的决策器。每一步只能做一件事：调用一个工具，或结束流程。\n"
    "规则：\n"
    "1. 必须先调用 check_consistency 和 check_coherence 完成两项校验；"
    "在任何一项校验执行之前，禁止结束流程，即使你认为对话没有问题。\n"
    "2. 校验报出违规后，先判断能否靠改 assistant 措辞修复：\n"
    "   - 答非所问（D1）、重复空转（D5）→ 调用 rewrite_segment 改写问题轮次；\n"
    "   - 改写后必须重新跑两个校验，确认问题消失。\n"
    "3. 涉及结构、实体、角色的问题（引用丢失、事实冲突、角色混淆、缺回复）无法靠改写措辞修复，"
    "直接调用 fallback_keep。\n"
    "4. 没有任何违规时结束流程（不要调用工具，直接给出简短理由）。\n"
    "5. 工具入参由运行时注入，你只需要选择工具名，不要自己拼参数。\n"
    "6. rewrite_segment 最多调用两次；两次后仍不通过就 fallback_keep。"
)

MAX_ISSUES_IN_PROMPT = 8
MAX_REASON_CHARS = 200
# 允许改写时的单对话重写上限：两次仍不通过就回退，避免空转烧 token
MAX_REWRITES = 2


class LLMPolicy:
    """function calling 决策策略：让模型选工具，与 RulePolicy 构成消融对照。

    设计取舍：**模型只选工具名，入参由运行时注入**。
    原因是让模型复述整段对话既贵又容易改坏内容（把"重构参数"当成"改写数据"），
    而这里要测的是"决策能力"，不是"抄写能力"。
    """

    def __init__(self, client: Any, *,
                 tools: Optional[Sequence[str]] = None,
                 max_tokens: int = 1024,
                 run_id: str = "adhoc",
                 allow_rewrite: bool = False):
        self.client = client
        self.tools = tuple(tools) if tools else None
        self.max_tokens = max_tokens
        self.run_id = run_id
        self.allow_rewrite = allow_rewrite
        self.system_prompt = REWRITE_SYSTEM_PROMPT if allow_rewrite else SYSTEM_PROMPT

    def decide(self, state: Any) -> Decision:
        resp = self.client.chat(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": self.observe(state)}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            tools=self._tool_specs(),
            stage="decide",
            run_id=self.run_id,
            dialogue_key=state.key,
        )
        state.add_tokens(resp.total_tokens)

        if resp.tool_calls:
            call = resp.tool_calls[0]
            name = call.get("name")
            if name not in self._tool_names():
                raise LLMPolicyError(f"模型选择了未注册的工具: {name!r}")
            if name in state.called_tools():
                if not self._repeat_allowed(name, state):
                    raise LLMPolicyError(f"模型重复调用工具: {name}")
            # 改写定位不到轮次时改走回退，而不是让整条样本 failed
            if name == "rewrite_segment" and RulePolicy._target_turn(state) is None:
                return Decision(
                    ACTION_CALL_TOOL, "fallback_keep",
                    {"original": state.original},
                    "违规无法定位到 assistant 轮次，回退原文",
                )
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
    def _repeat_allowed(self, name: str, state: Any) -> bool:
        """判定重复调用是否合法：只放行"状态已变、需要重看"的情形。

        - 改写会清空 issues 并换掉对话版本，之后重跑校验是必须的；
        - 只有成功的改写才解锁复检，失败的改写不能让模型无限重试。
        """
        history = list(state.history or [])
        if not history:
            return False

        if self.allow_rewrite and name == "rewrite_segment":
            used = sum(1 for h in history if h.get("tool") == name)
            return used < MAX_REWRITES

        if name in ("check_consistency", "check_coherence"):
            last = max(i for i, h in enumerate(history) if h.get("tool") == name)
            return any(
                h.get("tool") == "rewrite_segment" and h.get("ok", True)
                for h in history[last + 1:]
            )
        return False

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
        if tool == "rewrite_segment":
            turn = RulePolicy._target_turn(state)
            if turn is None:
                raise LLMPolicyError("rewrite_segment 无法定位到 assistant 轮次")
            return {
                "dialogue": state.dialogue,
                "turn": turn,
                "issues": list(state.issues),
            }
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
