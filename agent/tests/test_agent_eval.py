"""测试：Agent 臂评测聚合逻辑（全离线）

只验证"指标怎么算"，不验证模型好坏；真实数值由 benchmark 运行产出。
"""

import pytest

from agent.benchmark.agent_eval import (
    A3, A3L, A4, ArmResult, build_runtime, run_arm,
)
from agent.benchmark.defects import generate
from agent.runtime.llm import LLMResponse


class _CyclicClient:
    """交替返回两个校验工具调用，永不主动结束（用预算收口）。"""

    def __init__(self, tokens: int = 10):
        self.tokens = tokens
        self._i = 0

    def chat(self, messages, **kwargs):
        name = ("check_consistency", "check_coherence")[self._i % 2]
        self._i += 1
        return LLMResponse(
            text="",
            input_tokens=self.tokens,
            tool_calls=[{"name": name, "arguments": {}}],
        )


def _flag_all(_dialogue):
    return [{"rule": "C3_off_topic", "detail": "跑题", "turn": 1}]


@pytest.fixture
def dataset(mini_dialogues):
    defects = generate(mini_dialogues, per_type=2, seed=7)
    return mini_dialogues, defects


class TestBuildRuntime:
    def test_rule_arm_needs_no_client(self):
        assert build_runtime(A3) is not None

    def test_llm_arms_require_client(self):
        with pytest.raises(ValueError):
            build_runtime(A3L)
        with pytest.raises(ValueError):
            build_runtime(A4)

    def test_unknown_arm(self):
        with pytest.raises(ValueError):
            build_runtime("A9")


class TestRuleArm:
    def test_counts_and_block_rate(self, dataset):
        base, defects = dataset
        result = run_arm(A3, base, defects)

        assert result.arm == A3
        assert set(result.per_type) == {d["defect"]["type"] for d in defects}
        for row in result.per_type.values():
            assert row["total"] == 2

        # 结构类缺陷规则层必拦
        assert result.recall("D3_fact_conflict") == 1.0
        assert result.recall("D6_missing_response") == 1.0

    def test_clean_samples_not_flagged(self, dataset):
        base, defects = dataset
        result = run_arm(A3, base, defects)
        assert result.clean_total == len(base)
        assert result.clean_flagged == 0
        assert result.clean_fp_rate() == 0.0

    def test_rule_arm_spends_no_tokens(self, dataset):
        base, defects = dataset
        result = run_arm(A3, base, defects)
        assert result.tokens == 0
        assert result.cost_cny == 0.0


class TestLLMArmMetrics:
    def test_detected_and_blocked_are_distinct(self, dataset):
        """判官报违规但策略没回退 → detected 有、blocked 无。"""
        base, defects = dataset
        result = run_arm(
            A4, base, defects[:3],
            client=_CyclicClient(), judge=_flag_all,
        )
        for row in result.per_type.values():
            assert row["detected"] == row["total"], "判官每次都报违规"
            assert row["blocked"] == 0, "策略没调用 fallback_keep"
        assert result.clean_flagged == 0

    def test_tokens_are_accumulated(self, dataset):
        base, defects = dataset
        result = run_arm(A4, base, defects[:2], client=_CyclicClient(), judge=_flag_all)
        assert result.tokens > 0
        assert result.calls == 0, "没有 ledger 时不报调用次数"

    def test_policy_arm_without_judge(self, dataset):
        base, defects = dataset
        result = run_arm(A3L, base, defects[:2], client=_CyclicClient())
        assert result.arm == A3L
        assert result.tokens > 0
        assert all(row["detected"] == 0 for row in result.per_type.values())


class TestSerialization:
    def test_to_dict_fields(self):
        r = ArmResult(arm=A3, per_type={"D1_off_topic": {"total": 20, "detected": 0, "blocked": 20}},
                      clean_total=12, clean_flagged=0, tokens=100, cost_cny=0.25)
        d = r.to_dict()
        assert d["arm"] == A3
        assert d["macro_recall_blocked"] == 1.0
        assert d["clean_fp_rate"] == 0.0
        assert d["per_type"]["D1_off_topic"]["blocked"] == 20

    def test_macro_recall_averages_types(self):
        r = ArmResult(arm=A3, per_type={
            "D1": {"total": 10, "detected": 10, "blocked": 10},
            "D2": {"total": 10, "detected": 5, "blocked": 0},
        })
        assert r.macro_recall("blocked") == 0.5
        assert r.macro_recall("detected") == 0.75

    def test_empty_result_is_zero_not_crash(self):
        r = ArmResult(arm=A3)
        assert r.macro_recall() == 0.0
        assert r.clean_fp_rate() == 0.0
