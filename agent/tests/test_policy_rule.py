"""测试：RulePolicy 的确定性决策序列"""

import pytest

from agent.runtime.budget import Budget
from agent.runtime.policy import ACTION_CALL_TOOL, ACTION_STOP, RulePolicy
from agent.runtime.state import AgentState


def _state(dialogue, issues=None, history=None):
    return AgentState(
        dialogue=dialogue,
        original=dialogue,
        budget=Budget(),
        issues=list(issues or []),
        history=list(history or []),
    )


CHECKED = [{"tool": "check_consistency"}]
CLEAN_CHECK = CHECKED
FALLBACK_DONE = [{"tool": "check_consistency"}, {"tool": "fallback_keep"}]


class TestRulePolicy:
    def test_first_step_is_consistency_check(self, mini_dialogues):
        d = mini_dialogues[0]
        dec = RulePolicy().decide(_state(d))
        assert dec.action == ACTION_CALL_TOOL
        assert dec.tool == "check_consistency"
        assert dec.args["before"] == d and dec.args["after"] == d
        assert dec.reason

    def test_stops_when_clean(self, mini_dialogues):
        dec = RulePolicy().decide(_state(mini_dialogues[0], history=CLEAN_CHECK))
        assert dec.action == ACTION_STOP
        assert "接受" in dec.reason

    def test_falls_back_when_issues(self, mini_dialogues):
        dec = RulePolicy().decide(_state(
            mini_dialogues[0],
            issues=[{"rule": "R3_entity_fidelity", "detail": "实体丢失: X"}],
            history=CLEAN_CHECK,
        ))
        assert dec.action == ACTION_CALL_TOOL
        assert dec.tool == "fallback_keep"
        assert "1 处违规" in dec.reason

    def test_stops_after_fallback(self, mini_dialogues):
        dec = RulePolicy().decide(_state(
            mini_dialogues[0],
            issues=[{"rule": "R3_entity_fidelity", "detail": "x"}],
            history=FALLBACK_DONE,
        ))
        assert dec.action == ACTION_STOP

    def test_deterministic(self, mini_dialogues):
        """同一状态多次决策必须一致——基准可复现的前提。"""
        p = RulePolicy()
        s = _state(mini_dialogues[0], history=CLEAN_CHECK)
        assert p.decide(s) == p.decide(s)
