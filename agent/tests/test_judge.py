"""测试：语义判官（全离线，用 FakeLLMClient）

判官的价值在于 D1：规则层抓不到"答非所问"，字面相似度也只能碰运气。
这里验证的是契约与工程约束，不是判官准确率（准确率见真实测量）。
"""

import json

import pytest

from agent.eval.coherence import C0_JUDGE_ERROR, C3_OFF_TOPIC, check_coherence
from agent.eval.judge import (
    JUDGE_VERSION, JudgeError, LLMJudge, parse_verdict, render_dialogue,
    render_pairs,
)
from agent.runtime.llm import FakeLLMClient


def _dialogue(*contents, roles=None):
    roles = roles or ["user", "assistant"] * len(contents)
    return {"messages": [{"role": r, "content": c}
                         for r, c in zip(roles, contents)]}


def _verdicts(*items):
    """构造 v3 判官输出：(turn, related, detail) 三元组。"""
    return json.dumps(
        {"verdicts": [{"turn": t, "related": r, "detail": d}
                      for t, r, d in items]},
        ensure_ascii=False,
    )


def _judge_returning(text: str) -> LLMJudge:
    return LLMJudge(client=FakeLLMClient(default=text))


class TestRender:
    def test_includes_turn_index_and_role(self):
        text = render_dialogue(_dialogue("我想换货。", "好的，我帮您登记。"))
        assert "[0] user: 我想换货。" in text
        assert "[1] assistant: 好的，我帮您登记。" in text

    def test_pairs_bind_user_request_to_reply(self):
        text = render_pairs(_dialogue("我想换货。", "好的，我帮您登记。"))
        assert "user[0] 诉求：我想换货。" in text
        assert "assistant[1] 回复：好的，我帮您登记。" in text

    def test_truncates_long_content(self):
        text = render_dialogue(_dialogue("长" * 2000))
        assert len(text) < 2000
        assert "…" in text

    def test_empty_dialogue(self):
        assert render_dialogue({"messages": []}) == ""
        assert render_pairs({"messages": []}) == ""


class TestParseVerdict:
    def test_related_false_becomes_issue(self):
        d = _dialogue("我想问下物流。", "换货需要保持包装完整。")
        v = parse_verdict(_verdicts((1, False, "答的是换货")), d)
        assert [(x["rule"], x["turn"]) for x in v] == [(C3_OFF_TOPIC, 1)]
        assert v[0]["detail"] == "答的是换货"

    def test_all_related_true_is_clean(self):
        d = _dialogue("你好", "您好")
        assert parse_verdict(_verdicts((1, True, "")), d) == []

    def test_drops_turn_pointing_to_user(self):
        d = _dialogue("我想问下物流。", "换货需要保持包装完整。")
        text = _verdicts((0, False, ""), (1, True, ""))
        assert parse_verdict(text, d) == []

    def test_drops_out_of_range_turn(self):
        d = _dialogue("你好", "您好")
        text = _verdicts((99, False, ""), (1, True, ""))
        assert parse_verdict(text, d) == []

    def test_non_integer_turn_is_not_a_verdict(self):
        d = _dialogue("你好", "您好")
        with pytest.raises(JudgeError):
            parse_verdict('{"verdicts": [{"turn": "1", "related": false}]}', d)

    def test_default_detail(self):
        d = _dialogue("你好", "您好")
        v = parse_verdict(_verdicts((1, False, "")), d)
        assert v[0]["detail"]

    def test_invalid_json_raises(self):
        with pytest.raises(JudgeError):
            parse_verdict("我觉得都挺好的", _dialogue("你好", "您好"))

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_output_raises(self, text):
        """推理型模型可能把 max_tokens 全花在思考上，content 为空——必须显式失败。"""
        with pytest.raises(JudgeError, match="空内容"):
            parse_verdict(text, _dialogue("你好", "您好"))

    def test_non_object_raises(self):
        with pytest.raises(JudgeError):
            parse_verdict('["a"]', _dialogue("你好", "您好"))

    def test_missing_pair_is_judge_error(self):
        """漏判不是“没问题”：没验成的样本必须显式失败。"""
        d = _dialogue("你好", "您好")
        with pytest.raises(JudgeError, match="漏判"):
            parse_verdict(_verdicts((99, False, "")), d)

    def test_empty_verdicts_is_judge_error(self):
        d = _dialogue("你好", "您好")
        with pytest.raises(JudgeError, match="逐对判定"):
            parse_verdict('{"verdicts": []}', d)

    def test_conflicting_duplicate_turn_raises(self):
        d = _dialogue("你好", "您好")
        text = (
            '{"verdicts": [{"turn": 1, "related": true},'
            ' {"turn": 1, "related": false}]}'
        )
        with pytest.raises(JudgeError, match="冲突"):
            parse_verdict(text, d)

    def test_tolerates_repeated_top_level_keys(self):
        """实测 qwen2.5 会重复 "verdicts" 键；逐对提取应全部恢复。"""
        d = _dialogue("我想问下物流。", "换货需要保持包装完整。")
        text = (
            '{"verdicts": [{"turn": 1, "related": false, "detail": "跑题"}],'
            ' "verdicts": [{"turn": 1, "related": false, "detail": "跑题"}]}'
        )
        v = parse_verdict(text, d)
        assert [(x["turn"], x["detail"]) for x in v] == [(1, "跑题")]


class TestLLMJudge:
    def test_prompt_satisfies_json_mode(self):
        """DeepSeek 要求 prompt 里出现 json 字样，否则直接 400。"""
        client = FakeLLMClient(default=_verdicts((1, True, "")))
        LLMJudge(client=client)(_dialogue("你好", "您好"))
        joined = json.dumps(client.calls[0]["messages"], ensure_ascii=False).lower()
        assert "json" in joined

    def test_prompt_requests_per_pair_verdicts(self):
        client = FakeLLMClient(default=_verdicts((1, True, "")))
        LLMJudge(client=client)(_dialogue("你好", "您好"))
        joined = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
        assert "verdicts" in joined
        assert "related" in joined
        assert "user[0] 诉求" in joined
        assert "assistant[1] 回复" in joined

    def test_uses_zero_temperature(self):
        client = FakeLLMClient(default=_verdicts((1, True, "")))
        LLMJudge(client=client)(_dialogue("你好", "您好"))
        assert client.calls[0]["temperature"] == 0.0

    def test_reports_stage_and_dialogue_key(self):
        client = FakeLLMClient(default=_verdicts((1, True, "")))
        d = _dialogue("你好", "您好")
        LLMJudge(client=client, stage="judge", run_id="run-1")(d)
        call = client.calls[0]
        assert call["stage"] == "judge"
        assert call["run_id"] == "run-1"
        assert len(call["dialogue_key"]) == 16

    def test_verdict_carries_version(self):
        judge = _judge_returning(_verdicts((1, False, "跑题")))
        v = judge(_dialogue("我想问下物流。", "换货需要保持包装完整。"))
        assert v[0]["judge_version"] == JUDGE_VERSION

    def test_bad_output_raises(self):
        with pytest.raises(JudgeError):
            _judge_returning("我无法判断")(_dialogue("你好", "您好"))


class TestTokenAccounting:
    def test_tokens_used_accumulates(self):
        judge = LLMJudge(client=FakeLLMClient(default=_verdicts((1, True, ""))))
        d = _dialogue("你好", "您好")
        judge(d)
        first = judge.tokens_used
        assert first > 0
        judge(d)
        assert judge.tokens_used == first * 2


class TestIntegrationWithCheckCoherence:
    def test_judge_issue_blocks_dialogue(self):
        judge = _judge_returning(_verdicts((1, False, "答的是换货")))
        d = _dialogue("我想问下物流。", "换货需要保持包装完整。")
        rules = [v["rule"] for v in check_coherence(d, judge=judge)]
        assert rules == [C3_OFF_TOPIC]

    def test_broken_judge_does_not_pass_silently(self):
        judge = _judge_returning("not json")
        d = _dialogue("我想问下物流。", "换货需要保持包装完整。")
        rules = [v["rule"] for v in check_coherence(d, judge=judge)]
        assert rules == [C0_JUDGE_ERROR]
