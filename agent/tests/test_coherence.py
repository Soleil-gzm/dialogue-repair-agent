"""测试：连贯性检查（确定性部分 + 判官挂钩）

规则正确性在此保证；工具封装见 test_check_coherence，实测拦截率见 test_coherence_metrics。
"""

import copy

import pytest

from agent.eval.coherence import (
    C0_INPUT_CONTRACT, C0_JUDGE_ERROR, C1_REPETITION, C2_DEGENERATE_TURN, C3_OFF_TOPIC,
    check_coherence, check_degenerate, check_repetition, normalize_text, similarity,
)


def _dialogue(*contents, roles=None):
    roles = roles or ["assistant"] * len(contents)
    return {"messages": [{"role": r, "content": c} for r, c in zip(roles, contents)]}


class TestNormalize:
    def test_strips_whitespace_and_punctuation(self):
        assert normalize_text(" 已 为 您 查询，订单。 ") == "已为您查询订单"

    def test_similarity_of_identical_text_is_one(self):
        assert similarity("abc", "abc") == 1.0

    def test_similarity_of_unrelated_text_is_low(self):
        assert similarity("已为您查询订单正在分拣", "换货需要保持包装完整") < 0.3


class TestRepetition:
    def test_identical_adjacent_replies_flagged(self):
        d = _dialogue(
            "已为您查询，订单正在分拣中。",
            "我帮您登记换货。",
            "我帮您登记换货。",
        )
        v = check_repetition(d)
        assert len(v) == 1
        assert v[0]["rule"] == C1_REPETITION
        assert v[0]["turn"] == 2

    def test_punctuation_change_still_repetition(self):
        d = _dialogue("已为您查询，订单正在分拣中。", "已为您查询 订单正在分拣中")
        assert len(check_repetition(d)) == 1

    def test_paraphrase_not_flagged(self):
        """同话题换说法是正常对话，不能算重复空转。"""
        d = _dialogue("已为您查询，订单正在分拣中。", "物流进度我这边帮您盯着。")
        assert check_repetition(d) == []

    def test_non_adjacent_duplicate_not_flagged(self):
        """长对话里同句套话隔轮出现属正常现象，只比相邻轮。"""
        d = _dialogue("稍后会有短信通知，请留意。", "换货需要保持包装完整。",
                      "稍后会有短信通知，请留意。")
        assert check_repetition(d) == []

    def test_user_turns_ignored(self):
        d = _dialogue("同一句话", "同一句话", roles=["user", "user"])
        assert check_repetition(d) == []

    def test_empty_dialogue_ok(self):
        assert check_repetition({"messages": []}) == []


class TestDegenerate:
    @pytest.mark.parametrize("content", ["", "   ", "嗯", "。", "好。"])
    def test_empty_or_tiny_reply_flagged(self, content):
        v = check_degenerate(_dialogue("已为您查询，订单正在分拣中。", content))
        assert [x["rule"] for x in v] == [C2_DEGENERATE_TURN]
        assert v[0]["turn"] == 1

    def test_normal_reply_ok(self):
        assert check_degenerate(_dialogue("已为您查询，订单正在分拣中。")) == []


class TestJudgeHook:
    def test_judge_violations_are_merged(self):
        d = _dialogue("今天天气不错。", "您的退款会在三个工作日内到账。")
        v = check_coherence(d, judge=lambda _d: [
            {"rule": C3_OFF_TOPIC, "detail": "回复与上一轮问题无关", "turn": 1},
        ])
        assert [x["rule"] for x in v] == [C3_OFF_TOPIC]

    def test_judge_receives_dialogue(self):
        seen = {}

        def judge(d):
            seen["dialogue"] = d
            return []

        d = _dialogue("你好")
        check_coherence(d, judge=judge)
        assert seen["dialogue"] is d

    def test_judge_failure_is_not_silent_pass(self):
        """判官挂了要当违规：宁可回退，也不放行未经验证的候选。"""
        def judge(_d):
            raise RuntimeError("timeout")

        v = check_coherence(_dialogue("你好"), judge=judge)
        assert [x["rule"] for x in v] == [C0_JUDGE_ERROR]
        assert "timeout" in v[0]["detail"]

    def test_deterministic_rules_still_run_with_judge(self):
        d = _dialogue("同一句回复。", "同一句回复。")
        v = check_coherence(d, judge=lambda _d: [])
        assert [x["rule"] for x in v] == [C1_REPETITION]


class TestDedupe:
    def test_same_rule_same_turn_counted_once(self):
        d = _dialogue("同一句回复。", "同一句回复。")
        v = check_coherence(d, judge=lambda _d: [
            {"rule": C1_REPETITION, "detail": "判官也认为重复", "turn": 1},
        ])
        assert len(v) == 1
        assert v[0]["detail"].startswith("第 1 轮")

    def test_different_turns_kept(self):
        d = _dialogue("同一句回复。", "同一句回复。", "同一句回复。")
        v = check_coherence(d)
        assert {x["turn"] for x in v} == {1, 2}


class TestContract:
    def test_non_dict_input(self):
        v = check_coherence("not a dialogue")
        assert [x["rule"] for x in v] == [C0_INPUT_CONTRACT]

    def test_does_not_mutate_input(self):
        d = _dialogue("同一句回复。", "同一句回复。")
        before = copy.deepcopy(d)
        check_coherence(d)
        assert d == before
