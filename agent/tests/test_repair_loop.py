"""测试：多轮修复闭环

覆盖「修复成功 → 复检 → 接受」「重写失败/不可定位 → 回退」「重写次数上限」，
以及策略只依赖 history 做轮次判断这一契约。
"""

import copy
import random

from agent.benchmark.fixtures import make_dialogue
from agent.runtime.budget import Budget
from agent.runtime.loop import ACCEPTED, FALLBACK, STOPPED, AgentRuntime
from agent.runtime.policy import RulePolicy
from agent.runtime.registry import FunctionTool, ToolRegistry
from agent.runtime.state import AgentState
from agent.tools import register_all

REPAIRED = "已为您处理，请留意后续通知。"
# 多轮修复时每轮用不同句子，避免触发 C1 相邻复读规则
REPAIRED_BY_TURN = [
    "已为您处理，请留意后续通知。",
    "您反馈的问题我记录下来了，会转给对应同事。",
    "换货需要保持包装完整，我帮您登记。",
    "已为您查询，订单正在分拣中。",
    "很抱歉给您带来不便，我这边同步处理。",
    "稍后会有短信通知，请留意。",
]


def _fake_rewrite(fix=REPAIRED, fail: bool = False) -> FunctionTool:
    def _run(state, args):
        turn = args.get("turn")
        if fail:
            return {
                "ok": False,
                "violations": [{"rule": "W1_parse_error",
                                "detail": "模型输出不可用", "turn": turn}],
                "tokens": 1,
            }
        text = fix(turn) if callable(fix) else fix
        dialogue = copy.deepcopy(args["dialogue"])
        dialogue["messages"][turn]["content"] = text
        return {
            "ok": True,
            "status": "rewritten",
            "dialogue": dialogue,
            "rewritten_turn": turn,
            "tokens": 1,
        }

    return FunctionTool(
        name="rewrite_segment",
        description="测试用假重写",
        schema={"type": "object", "properties": {}},
        fn=_run,
    )


def _setup(tool: FunctionTool) -> None:
    ToolRegistry.clear()
    register_all()
    ToolRegistry.register(tool)


def _budget(iterations: int = 20, calls: int = 24) -> Budget:
    return Budget(max_iterations=iterations, max_tool_calls=calls,
                  max_tokens=20000, max_wall_clock_s=300.0)


def _dialogue(n_turns: int = 1, blank: bool = True) -> dict:
    d = make_dialogue(4, n_turns, random.Random(7))
    if blank:
        for msg in d["messages"]:
            if msg.get("role") == "assistant":
                msg["content"] = ""
    return d


def _first_assistant(dialogue: dict) -> int:
    return next(i for i, m in enumerate(dialogue["messages"])
                if m.get("role") == "assistant")


def _runtime(tool: FunctionTool) -> AgentRuntime:
    _setup(tool)
    return AgentRuntime(RulePolicy(rewrite_tool="rewrite_segment"), budget=_budget())


def _rewrite_calls(result) -> int:
    return sum(1 for e in result.trace
               if e.get("event") == "tool_result" and e.get("tool") == "rewrite_segment")


class TestRepairLoop:
    def test_repairs_then_accepts(self):
        dialogue = _dialogue(1)
        before = copy.deepcopy(dialogue)
        result = _runtime(_fake_rewrite()).run(dialogue)

        assert result.status == ACCEPTED
        assert result.issues == []
        assert result.seen_issues  # 曾经检出过，但已修复
        assert result.dialogue["messages"][_first_assistant(dialogue)]["content"] == REPAIRED
        assert result.tool_calls == 5  # 2 校验 + 1 重写 + 2 复检
        assert dialogue == before  # 输入不被原地修改

    def test_failed_rewrite_falls_back(self):
        dialogue = _dialogue(1)
        result = _runtime(_fake_rewrite(fail=True)).run(dialogue)

        assert result.status == FALLBACK
        assert result.dialogue == dialogue
        assert result.issues  # 原始违规仍保留
        assert _rewrite_calls(result) == 1

    def test_same_turn_limit_then_fallback(self):
        dialogue = _dialogue(1)
        # "好" 只有 1 字，仍低于 MIN_ASSISTANT_CHARS，复检必然继续失败
        result = _runtime(_fake_rewrite(fix="好")).run(dialogue)

        assert result.status == FALLBACK
        assert _rewrite_calls(result) == 2

    def test_total_round_limit_then_fallback(self):
        dialogue = _dialogue(6)
        fix = lambda turn: REPAIRED_BY_TURN[turn // 2 % len(REPAIRED_BY_TURN)]
        result = _runtime(_fake_rewrite(fix=fix)).run(dialogue)

        assert result.status == FALLBACK
        assert _rewrite_calls(result) == 4

    def test_budget_exhausted_after_rewrite_is_not_accepted(self):
        _setup(_fake_rewrite())
        runtime = AgentRuntime(
            RulePolicy(rewrite_tool="rewrite_segment"),
            budget=_budget(iterations=3, calls=6),
        )
        result = runtime.run(_dialogue(1))

        assert result.status == STOPPED
        assert result.termination == "max_iterations"


class TestRepairPolicy:
    def _state(self, history, issues, dialogue=None):
        dialogue = dialogue or _dialogue(1)
        return AgentState(dialogue=dialogue, original=copy.deepcopy(dialogue),
                          budget=Budget(), history=history, issues=issues)

    def test_unlocatable_issue_falls_back(self):
        state = self._state(
            [{"tool": "check_consistency", "ok": True},
             {"tool": "check_coherence", "ok": False}],
            [{"rule": "R3_entity_fidelity", "detail": "实体丢失", "turn": None}],
        )
        decision = RulePolicy(rewrite_tool="rewrite_segment").decide(state)
        assert decision.tool == "fallback_keep"

    def test_failed_rewrite_falls_back_without_recheck(self):
        state = self._state(
            [{"tool": "check_consistency", "ok": True},
             {"tool": "check_coherence", "ok": False},
             {"tool": "rewrite_segment", "ok": False, "turn": 2}],
            [{"rule": "C2_degenerate_turn", "detail": "过短", "turn": 2}],
        )
        decision = RulePolicy(rewrite_tool="rewrite_segment").decide(state)
        assert decision.tool == "fallback_keep"

    def test_rechecks_after_successful_rewrite(self):
        state = self._state(
            [{"tool": "check_consistency", "ok": True},
             {"tool": "check_coherence", "ok": False},
             {"tool": "rewrite_segment", "ok": True, "turn": 2}],
            [],
        )
        decision = RulePolicy(rewrite_tool="rewrite_segment").decide(state)
        assert decision.tool == "check_consistency"
