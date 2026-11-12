"""测试：缺陷注入集与 M1 门禁指标

M1 门禁：
- 一致性拦截率 ≥ 99%（D3 事实冲突必须全部拦住）
- 误杀率 ≤ 5%（干净样本不得被误判）
"""

import copy
import json

import pytest

from agent.benchmark.defects import (
    DEFECT_TYPES, OFF_TOPIC_REPLIES, generate, inject, to_jsonl,
)
from agent.benchmark.fixtures import generate_dialogues
from agent.eval.rules import check_consistency


@pytest.fixture
def dialogues():
    return generate_dialogues(12, seed=20261112)


class TestGenerate:
    def test_covers_all_types(self, dialogues):
        records = generate(dialogues, per_type=5, seed=7)
        assert len(records) == len(DEFECT_TYPES) * 5
        assert {r["defect"]["type"] for r in records} == set(DEFECT_TYPES)

    def test_sample_keys_unique(self, dialogues):
        records = generate(dialogues, per_type=5, seed=7)
        keys = [r["sample_key"] for r in records]
        assert len(keys) == len(set(keys))

    def test_deterministic_same_seed(self, dialogues):
        a = to_jsonl(generate(dialogues, per_type=3, seed=42))
        b = to_jsonl(generate(dialogues, per_type=3, seed=42))
        assert a == b

    def test_different_seed_differs(self, dialogues):
        a = to_jsonl(generate(dialogues, per_type=3, seed=1))
        b = to_jsonl(generate(dialogues, per_type=3, seed=2))
        assert a != b

    def test_input_not_mutated(self, dialogues):
        before = json.dumps(dialogues, sort_keys=True, ensure_ascii=False)
        generate(dialogues, per_type=2, seed=7)
        assert json.dumps(dialogues, sort_keys=True, ensure_ascii=False) == before

    def test_single_dialogue_is_enough(self):
        """D1 不再依赖 donor 对话，单条样本也能生成全部类型。"""
        records = generate(dialogues=[generate_dialogues(1, seed=1)[0]], per_type=1)
        assert len(records) == len(DEFECT_TYPES)

    def test_unknown_type_raises(self, dialogues):
        import random
        with pytest.raises(ValueError):
            inject(dialogues[0], "D9_nope", random.Random(0))

    def test_jsonl_roundtrip(self, dialogues):
        records = generate(dialogues, per_type=2, seed=7)
        lines = to_jsonl(records).strip().split("\n")
        assert len(lines) == len(records)
        assert json.loads(lines[0])["sample_key"] == records[0]["sample_key"]


class TestInjectionShapes:
    def test_d1_replaces_with_cross_topic_reply(self, dialogues):
        import random
        bad, meta = inject(dialogues[0], "D1_off_topic", random.Random(1))
        assert meta.get("skipped") is not True
        turn = meta["turn"]
        injected = bad["messages"][turn]["content"]
        assert injected in OFF_TOPIC_REPLIES
        assert injected != dialogues[0]["messages"][turn]["content"], "注入不能是空操作"
        # 与 user 内容相同会变成 D4（角色混淆），两类的判定必须可分
        assert injected not in {m["content"] for m in bad["messages"]
                                if m["role"] == "user"}

    def test_d2_actually_strips_id(self, dialogues):
        import random
        bad, meta = inject(dialogues[0], "D2_reference_broken", random.Random(1))
        assert meta.get("skipped") is not True, "D2 未能注入：样本里没有可删除的编号"
        assert "ORD-" not in bad["messages"][meta["turn"]]["content"]

    def test_d3_adds_amount(self, dialogues):
        import random
        bad, meta = inject(dialogues[0], "D3_fact_conflict", random.Random(1))
        assert meta.get("skipped") is not True
        assert "¥" in bad["messages"][meta["turn"]]["content"]

    def test_d6_removes_a_message(self, dialogues):
        import random
        bad, meta = inject(dialogues[0], "D6_missing_response", random.Random(1))
        assert meta.get("skipped") is not True
        assert len(bad["messages"]) == len(dialogues[0]["messages"]) - 1

    def test_d4_puts_user_text_into_assistant(self, dialogues):
        import random
        bad, meta = inject(dialogues[0], "D4_role_confusion", random.Random(1))
        assert meta.get("skipped") is not True
        assert bad["messages"][meta["turn"]]["role"] == "assistant"


class TestM1Gate:
    def test_fact_conflict_blocked_100pct(self, dialogues):
        """门禁：D3 事实冲突必须 100% 被 check_consistency 拦截（要求 ≥99%）。"""
        records = [r for r in generate(dialogues, per_type=20, seed=7)
                   if r["defect"]["type"] == "D3_fact_conflict"]
        base = [dialogues[r["base_index"]] for r in records]
        blocked = sum(1 for r, b in zip(records, base)
                      if check_consistency(b, r["dialogue"]))
        rate = blocked / len(records)
        assert rate >= 0.99, f"拦截率 {rate:.2%} 低于门禁 99%"

    def test_missing_response_blocked(self, dialogues):
        records = [r for r in generate(dialogues, per_type=10, seed=7)
                   if r["defect"]["type"] == "D6_missing_response"
                   and not r["defect"].get("skipped")]
        base = [dialogues[r["base_index"]] for r in records]
        blocked = sum(1 for r, b in zip(records, base)
                      if check_consistency(b, r["dialogue"]))
        assert blocked == len(records)

    def test_false_positive_rate_zero(self, dialogues):
        """门禁：干净样本误杀率 0%（要求 ≤5%）。"""
        rejected = sum(1 for d in dialogues
                       if check_consistency(d, copy.deepcopy(d)))
        assert rejected == 0

    def test_repetition_out_of_scope_for_consistency(self, dialogues):
        """诚实标注范围：D5 重复空转属连贯性问题，不该由一致性规则拦。"""
        records = [r for r in generate(dialogues, per_type=10, seed=7)
                   if r["defect"]["type"] == "D5_repetition"
                   and not r["defect"].get("skipped")]
        assert records, "应至少产生一条 D5 样本"
        # 结构、system、实体均未变 —— 一致性检查通过，留给 check_coherence（M2）
        for r in records:
            assert check_consistency(dialogues[r["base_index"]], r["dialogue"]) == []
