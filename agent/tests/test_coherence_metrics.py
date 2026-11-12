"""M3 门禁：连贯性检查的拦截率与误杀率

数据：agent/fixtures/public/ 下已入库的公开轨
- base_v1.json      12 条干净对话
- defects_v1.jsonl  120 条注入缺陷（6 类 × 20）

诚实边界：
- D5 重复空转是**字面复制**，确定性检查可全部拦下。
- D1 答非所问必须理解语义，确定性检查只能顺带命中"复制了邻轮内容"的样本，
  真实数据上不可依赖；D1 的召回要靠注入的语义判官（M3 后续步骤）。

断言"D1 未被全部拦下"是故意的：防止有人把字面相似度误当成语义理解。
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from agent.eval.coherence import check_coherence

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "public"
BASE = FIXTURES / "base_v1.json"
DEFECTS = FIXTURES / "defects_v1.jsonl"

BLOCK_RATE_MIN = 0.99
FALSE_REJECT_MAX = 0.05


@pytest.fixture(scope="module")
def base():
    return json.loads(BASE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def defects():
    lines = DEFECTS.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]


def _blocked(dialogue: dict) -> bool:
    return bool(check_coherence(dialogue))


class TestRepetitionRecall:
    def test_d5_blocked_100pct(self, defects):
        samples = [r for r in defects if r["defect"]["type"] == "D5_repetition"]
        hit = sum(1 for r in samples if _blocked(r["dialogue"]))
        assert hit / len(samples) >= BLOCK_RATE_MIN, f"D5 拦截 {hit}/{len(samples)}"

    def test_d5_violation_names_the_turn(self, defects):
        r = next(r for r in defects
                 if r["defect"]["type"] == "D5_repetition" and _blocked(r["dialogue"]))
        v = check_coherence(r["dialogue"])
        assert v[0]["rule"] == "C1_assistant_repetition"
        assert v[0]["turn"] == r["defect"]["turn"]


class TestFalsePositive:
    def test_clean_samples_not_flagged(self, base):
        flagged = [i for i, d in enumerate(base) if _blocked(d)]
        assert len(flagged) / len(base) <= FALSE_REJECT_MAX, f"误报: {flagged}"

    def test_other_defect_types_not_flagged(self, defects):
        """D2/D3/D4/D6 是保真类缺陷，不该被连贯性检查误报。"""
        others = [r for r in defects
                  if r["defect"]["type"] in
                  {"D2_reference_broken", "D3_fact_conflict",
                   "D4_role_confusion", "D6_missing_response"}]
        flagged = [r["sample_key"] for r in others if _blocked(r["dialogue"])]
        assert len(flagged) / len(others) <= FALSE_REJECT_MAX, f"误报: {flagged[:5]}"


class TestCapabilityBoundary:
    def test_d1_needs_semantic_judge(self, base, defects):
        """字面检查抓不全 D1：命中只来自"复制了邻轮内容"的巧合。"""
        samples = [r for r in defects if r["defect"]["type"] == "D1_off_topic"]
        hit = sum(1 for r in samples if _blocked(r["dialogue"]))
        assert hit < len(samples), "D1 被全部拦下，请复核是否已引入语义判官"

    def test_no_semantic_claim_without_judge(self):
        """没有判官时，语义类违规（C3）不可能出现。"""
        rules = Counter()
        for r in json.loads(BASE.read_text(encoding="utf-8")):
            rules.update(v["rule"] for v in check_coherence(r))
        assert "C3_off_topic" not in rules
