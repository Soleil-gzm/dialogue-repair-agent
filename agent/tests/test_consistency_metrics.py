"""M1 门禁：规则层拦截率与误杀率

数据来源：agent/fixtures/public/ 下两个已入库的公开轨文件
- base_v1.json      12 条干净对话
- defects_v1.jsonl  120 条注入缺陷（6 类 × 20）

诚实边界：规则层只能抓"结构性"缺陷（丢实体、改结构、换角色）。
语义类缺陷（答非所问、重复空转）需要 M2 的语义检查，这里不做断言，
只在 `test_semantic_defects_need_semantic_check` 中记录当前能力边界。
"""

import json
from pathlib import Path

import pytest

from agent.eval.rules import check_consistency

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "public"
BASE = FIXTURES / "base_v1.json"
DEFECTS = FIXTURES / "defects_v1.jsonl"

# 规则层应当稳定拦截的类型
RULE_DETECTABLE = {
    "D2_reference_broken",   # 删实体 → R3
    "D3_fact_conflict",      # 增实体 → R3
    "D4_role_confusion",     # assistant 复读 user 内容 → R5
    "D6_missing_response",   # 删消息 → R2
}

# 需要语义检查才能发现的类型
SEMANTIC_ONLY = {"D1_off_topic", "D5_repetition"}

BLOCK_RATE_MIN = 0.99
FALSE_REJECT_MAX = 0.05


@pytest.fixture(scope="module")
def base():
    return json.loads(BASE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def defects():
    lines = DEFECTS.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]


def _blocked(before: dict, after: dict) -> bool:
    return len(check_consistency(before, after)) > 0


class TestFixtureIntegrity:
    def test_defect_count(self, defects):
        assert len(defects) == 120

    def test_six_types_each_20(self, defects):
        from collections import Counter
        c = Counter(r["defect"]["type"] for r in defects)
        assert len(c) == 6
        assert all(v == 20 for v in c.values())

    def test_samples_are_actually_defective(self, defects):
        """注入不能是空操作：每条样本都必须与原文不同。"""
        base = json.loads(BASE.read_text(encoding="utf-8"))
        identical = [
            r["sample_key"] for r in defects
            if r["dialogue"] == base[r["base_index"]]
        ]
        assert not identical, f"注入无效（与原文相同）: {identical[:5]}"


class TestBlockRate:
    def test_rule_detectable_block_rate(self, base, defects):
        """门禁：结构性缺陷拦截率 ≥ 99%。"""
        total = hit = 0
        missed = []
        for r in defects:
            if r["defect"]["type"] not in RULE_DETECTABLE:
                continue
            total += 1
            if _blocked(base[r["base_index"]], r["dialogue"]):
                hit += 1
            else:
                missed.append(r["sample_key"])
        rate = hit / total
        assert rate >= BLOCK_RATE_MIN, (
            f"拦截率 {rate:.3f} < {BLOCK_RATE_MIN}；漏检样例: {missed[:5]}"
        )

    def test_clean_pairs_not_rejected(self, base):
        """门禁：误杀率 ≤ 5%（原文与自身比较必须通过）。"""
        rejected = [i for i, d in enumerate(base) if _blocked(d, d)]
        rate = len(rejected) / len(base)
        assert rate <= FALSE_REJECT_MAX, f"误杀率 {rate:.3f}；被误杀: {rejected}"


class TestCapabilityBoundary:
    def test_semantic_defects_need_semantic_check(self, base, defects):
        """记录能力边界：语义类缺陷规则层基本抓不到，留给 M2。

        这不是缺陷，而是设计边界——断言"抓不到"可防止有人误以为规则层足够。
        """
        total = hit = 0
        for r in defects:
            if r["defect"]["type"] not in SEMANTIC_ONLY:
                continue
            total += 1
            if _blocked(base[r["base_index"]], r["dialogue"]):
                hit += 1
        # 只要求"不是全部能抓"，避免把能力边界误报成能力
        assert hit < total, (
            f"语义类缺陷被规则层全部拦截（{hit}/{total}）——"
            "请复核 RULE_DETECTABLE 的分类是否已过时"
        )
