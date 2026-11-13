"""M2 门禁：决策循环

覆盖四件事：
1. 正常路径能收敛（校验 → 接受 / 回退）
2. 预算触顶即停，不死循环
3. 重复决策被识别为死循环
4. 决策链完整（每次工具调用都有对应 Trace）
"""

import copy
import random

import pytest

from agent.benchmark.defects import inject
from agent.runtime.budget import Budget
from agent.runtime.loop import (
    ACCEPTED, FALLBACK, FAILED, STOPPED, AgentRuntime,
    TERM_LOOP_DETECTED, TERM_STOP, TERM_TOOL_TERMINAL,
)
from agent.runtime.policy import ACTION_CALL_TOOL, ACTION_STOP, Decision, RulePolicy
from agent.runtime.registry import FunctionTool, ToolRegistry
from agent.runtime.trace import TraceWriter
from agent.tools import register_all


@pytest.fixture(autouse=True)
def tools_registered():
    register_all()


class _AlwaysRepeatPolicy:
    """永远返回同一个调用，用来验证死循环识别。"""

    def decide(self, state):
        return Decision(ACTION_CALL_TOOL, "check_consistency",
                        {"before": state.original, "after": state.dialogue}, "重复")


class _RaisingPolicy:
    def decide(self, state):
        raise RuntimeError("策略炸了")


class TestHappyPath:
    def test_clean_dialogue_accepted(self, mini_dialogues):
        r = AgentRuntime(RulePolicy()).run(mini_dialogues[0])
        assert r.status == ACCEPTED
        assert r.termination == TERM_STOP
        assert r.tool_calls == 2  # 保真校验 + 连贯性检查
        assert r.iterations == 2
        assert r.issues == []

    def test_dialogue_unchanged_when_accepted(self, mini_dialogues):
        before = copy.deepcopy(mini_dialogues[0])
        r = AgentRuntime(RulePolicy()).run(mini_dialogues[0])
        assert r.dialogue == before


class TestFallbackPath:
    def test_defective_dialogue_falls_back(self, mini_dialogues):
        clean = mini_dialogues[0]
        bad, _ = inject(clean, "D3_fact_conflict", random.Random(1))
        r = AgentRuntime(RulePolicy()).run(bad, reference=clean)
        assert r.status == FALLBACK
        assert r.termination == TERM_TOOL_TERMINAL
        assert r.tool_calls == 3  # 两个校验 + 回退
        assert r.issues, "应记录违规明细"

    def test_fallback_returns_reference_exactly(self, mini_dialogues):
        clean = mini_dialogues[0]
        bad, _ = inject(clean, "D3_fact_conflict", random.Random(1))
        r = AgentRuntime(RulePolicy()).run(bad, reference=clean)
        assert r.dialogue == clean, "回退目标是原始参照，不是受损的候选"


class TestBudgetGuards:
    def test_stops_when_iterations_exhausted(self, mini_dialogues):
        r = AgentRuntime(RulePolicy(), budget=Budget(max_iterations=0)).run(mini_dialogues[0])
        assert r.status == STOPPED
        assert r.tool_calls == 0

    def test_stops_when_tool_calls_exhausted(self, mini_dialogues):
        clean = mini_dialogues[0]
        bad, _ = inject(clean, "D6_missing_response", random.Random(1))
        r = AgentRuntime(RulePolicy(), budget=Budget(max_tool_calls=1)).run(bad, reference=clean)
        assert r.status == STOPPED
        assert r.tool_calls == 1


def _burning_tool(tokens: int = 500) -> FunctionTool:
    return FunctionTool(
        name="burn_tokens",
        description="测试用：只上报 token 消耗",
        schema={"type": "object", "properties": {}},
        fn=lambda _state, _args: {"ok": True, "tokens": tokens},
    )


class _BurningPolicy:
    """每次决策自报一定 token 用量，用来验证预算是真的在管。"""

    def __init__(self, tokens_per_decide: int = 0):
        self.tokens = tokens_per_decide

    def decide(self, state):
        state.add_tokens(self.tokens)
        return Decision(ACTION_CALL_TOOL, "burn_tokens", {}, "花 token")


class TestTokenBudget:
    def test_tool_reported_tokens_exhaust_budget(self, mini_dialogues):
        ToolRegistry.register(_burning_tool(500))
        r = AgentRuntime(_BurningPolicy(), budget=Budget(max_tokens=100)).run(mini_dialogues[0])
        assert r.status == STOPPED
        assert r.tokens == 500

    def test_policy_tokens_exhaust_budget(self, mini_dialogues):
        """策略自己调模型的消耗也要占预算，否则上限形同虚设。"""
        ToolRegistry.register(_burning_tool(0))
        r = AgentRuntime(_BurningPolicy(500), budget=Budget(max_tokens=100)).run(mini_dialogues[0])
        assert r.status == STOPPED
        assert r.tokens == 500
        assert r.tool_calls == 1, "第一次调用后即应触顶，不再有第二次"

    def test_stop_decision_still_reports_usage(self, mini_dialogues):
        class _SpendThenStop:
            def decide(self, state):
                state.add_tokens(120)
                return Decision(ACTION_STOP, reason="结束")

        r = AgentRuntime(_SpendThenStop()).run(mini_dialogues[0])
        assert r.status == ACCEPTED
        assert r.tokens == 120

    def test_drain_is_one_shot(self, mini_dialogues):
        from agent.runtime.state import AgentState

        s = AgentState(dialogue=mini_dialogues[0], original=mini_dialogues[0])
        s.add_tokens(10)
        s.add_tokens(5)
        assert s.drain_tokens() == 15
        assert s.drain_tokens() == 0


class TestLoopDetection:
    def test_repeated_decision_detected(self, mini_dialogues):
        r = AgentRuntime(_AlwaysRepeatPolicy()).run(mini_dialogues[0])
        assert r.termination == TERM_LOOP_DETECTED
        # 第二次重复即终止，不会跑满预算
        assert r.tool_calls <= 2


class TestErrorIsolation:
    def test_policy_exception_marks_failed(self, mini_dialogues):
        r = AgentRuntime(_RaisingPolicy()).run(mini_dialogues[0])
        assert r.status == FAILED
        assert "策略异常" in r.reason


class TestTraceCompleteness:
    def test_every_tool_call_has_trace(self, mini_dialogues):
        clean = mini_dialogues[0]
        bad, _ = inject(clean, "D3_fact_conflict", random.Random(1))
        r = AgentRuntime(RulePolicy()).run(bad, reference=clean)
        results = [x for x in r.trace if x["event"] == "tool_result"]
        assert len(results) == r.tool_calls

    def test_trace_has_start_and_end(self, mini_dialogues):
        r = AgentRuntime(RulePolicy()).run(mini_dialogues[0])
        events = [x["event"] for x in r.trace]
        assert events[0] == "start"
        assert events[-1] == "end"

    def test_every_decision_has_reason(self, mini_dialogues):
        r = AgentRuntime(RulePolicy()).run(mini_dialogues[0])
        for rec in r.trace:
            if rec["event"] == "decide":
                assert rec["reason"], "决策必须给出理由"

    def test_trace_written_to_disk(self, mini_dialogues, tmp_path):
        path = tmp_path / "trace.jsonl"
        with TraceWriter(path) as tw:
            AgentRuntime(RulePolicy(), trace=tw).run(mini_dialogues[0])
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 4  # start + decide + tool_result + end
