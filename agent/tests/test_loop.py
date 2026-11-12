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
from agent.runtime.policy import ACTION_CALL_TOOL, Decision, RulePolicy
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
