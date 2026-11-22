"""测试：退化指标（纯函数 + 接入 arm / compare 结果）"""

import pytest

from agent.benchmark.agent_eval import A3, compare_arms, run_arm
from agent.benchmark.defects import generate
from agent.benchmark.degradation import (
    MAX_DISTINCT_DROP, MAX_REPEAT_RISE,
    assistant_text, cross_sample_repeat_rate, distinct_n, length_ratio, measure,
)


def _dialogue(*assistant_contents):
    """构造对话：每个参数都作为一条 assistant 回复，附带固定 system 提示。"""
    messages = [{"role": "system", "content": "系统提示"}]
    for text in assistant_contents:
        messages.append({"role": "user", "content": "用户问题"})
        messages.append({"role": "assistant", "content": text})
    return {"messages": messages}


class TestPureFunctions:
    def test_only_assistant_text_is_measured(self):
        d = {"messages": [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "用户问题"},
            {"role": "assistant", "content": "客服回复"},
            {"role": "user", "content": "追问"},
        ]}
        assert assistant_text(d) == "客服回复"

    def test_length_ratio_is_corpus_level(self):
        assert length_ratio(["ab", "cd"], ["abcd"]) == pytest.approx(1.0)
        assert length_ratio([], []) is None
        assert length_ratio(["", " "], ["x"]) is None

    def test_distinct_n_counts_unique_grams(self):
        # "abab" 的 bigram：ab, ba, ab → 唯一 2 个 / 总 3 个
        assert distinct_n(["abab"], 2) == pytest.approx(2 / 3)
        assert distinct_n([], 2) is None

    def test_cross_sample_repeat_rate(self):
        # 两条互不相干的短文本：没有共享 4-gram
        assert cross_sample_repeat_rate(["abcdefgh", "ijklmnop"], 4) == 0.0
        # 改写后洗成同一句：全部 4-gram 被两条共用
        assert cross_sample_repeat_rate(["abcdefgh", "abcdefgh"], 4) == 1.0

    def test_identical_dialogue_passes_all(self):
        d = _dialogue("问题", "一段足够长的客服回复内容")
        result = measure([d], [d])
        assert result["n_samples"] == 1
        assert result["length_ratio"]["ratio"] == 1.0
        assert result["degraded"] is False
        assert result["length_ratio"]["verdict"] == "pass"


class TestDegradationVerdicts:
    def test_length_inflation_fails(self):
        before = _dialogue("问题", "短回复")
        after = _dialogue("问题", "短回复" * 4)
        result = measure([before], [after])
        assert result["length_ratio"]["ratio"] > 1.3
        assert result["length_ratio"]["verdict"] == "fail"
        assert result["degraded"] is True

    def test_distinct_drop_fails(self):
        before = [_dialogue("问题", f"回复内容{i}各不相同{i}") for i in range(4)]
        after = [_dialogue("问题", "完全相同的客服话术完全相同的客服话术") for _ in range(4)]
        result = measure(before, after)
        assert result["distinct_2"]["change"] < -MAX_DISTINCT_DROP
        assert result["distinct_2"]["verdict"] == "fail"

    def test_cross_sample_repeat_rise_fails(self):
        before = [_dialogue("问题", "甲乙丙丁戊己庚辛"), _dialogue("问题", "子丑寅卯辰巳午未")]
        after = [_dialogue("问题", "同样一句客服话术"), _dialogue("问题", "同样一句客服话术")]
        result = measure(before, after)
        assert result["cross_sample_4gram"]["before"] == 0.0
        assert result["cross_sample_4gram"]["change"] is None
        # 0 → 正数没有比值，但必须判为退化，不能报 n/a 放过
        assert result["cross_sample_4gram"]["verdict"] == "fail"
        assert result["cross_sample_4gram"]["after"] == 1.0

    def test_unrelated_rewrite_within_threshold_passes(self):
        before = [_dialogue("问题", f"旧回复第{i}版内容") for i in range(3)]
        after = [_dialogue("问题", f"新回复第{i}版内容") for i in range(3)]
        result = measure(before, after)
        assert result["degraded"] is False

    def test_empty_measure_is_not_degraded(self):
        result = measure([], [])
        assert result["n_samples"] == 0
        assert result["degraded"] is False
        assert result["length_ratio"]["verdict"] == "n/a"

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            measure([_dialogue("u", "a")], [])


class TestWiring:
    def test_run_arm_reports_degradation(self, mini_dialogues):
        defects = generate(mini_dialogues, per_type=1, seed=3)
        result = run_arm(A3, mini_dialogues, defects)
        deg = result.to_dict()["degradation"]
        assert deg["n_samples"] == len(defects) + len(mini_dialogues)
        assert "length_ratio" in deg and "distinct_2" in deg

    def test_compare_arms_reports_degradation_for_both_arms(self, mini_dialogues):
        defects = generate(mini_dialogues, per_type=1, seed=3)
        payload = compare_arms(A3, A3, mini_dialogues, defects)
        assert set(payload["degradation"]) == {A3}
        assert payload["degradation"][A3]["n_samples"] > 0
        # A3 对 A3：outcome 完全一致，胜率不可比
        assert payload["win_rate"]["point"] is None
