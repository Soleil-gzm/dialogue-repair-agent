"""测试：公开轨分布校准档

校准档只允许匹配私有轨的聚合形状，不能复制原文。门禁分三类：
1. v1 冻结产物不变；
2. realistic 档的轮次、重复率、长度、实体密度落在目标区间；
3. 校准档仍可支撑六类缺陷注入与 dev/test 切分。
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

import pytest

from agent.benchmark import profiles
from agent.benchmark.defects import DEFECT_TYPES, generate, is_skipped
from agent.benchmark.fixtures import generate_dialogues, generate_split
from agent.benchmark.realistic import fixture_stats

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "public"
_ID_PATTERN = re.compile(r"\b[A-Z]{2,6}-\d{3,}\b")
_AMOUNT_PATTERN = re.compile(r"\d+(?:\.\d{2})")


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _turns(dialogue: dict) -> int:
    return sum(1 for m in dialogue["messages"] if m.get("role") == "assistant")


def _roles(dialogue: dict):
    return [m.get("role") for m in dialogue["messages"]]


def _jacobian(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


class TestFrozenV1:
    def test_default_output_matches_committed_base(self):
        assert generate_dialogues(12, 20261112) == _load("base_v1.json")

    def test_unknown_profile_rejected(self):
        with pytest.raises(ValueError):
            generate_dialogues(1, 1, profile="unknown")


class TestRealisticCalibration:
    @pytest.fixture(scope="class")
    def data(self):
        return generate_dialogues(1000, profiles.REALISTIC_BASE_SEED,
                                  profile=profiles.PROFILE_REALISTIC)

    def test_turn_histogram_matches_private_aggregate(self, data):
        counts = Counter(_turns(d) for d in data)
        assert dict(sorted(counts.items())) == profiles.REALISTIC_TURN_HISTOGRAM
        sizes = [len(d["messages"]) for d in data]
        assert (min(sizes), max(sizes)) == profiles.REALISTIC_MESSAGE_RANGE

    def test_repetition_rates_match_private_aggregate(self, data):
        stats = fixture_stats(data)
        assert 0.45 <= stats["assistant_unique_rate"] <= 0.55
        assert 0.24 <= stats["user_unique_rate"] <= 0.34
        assert 0.02 <= stats["internal_dup_rate"] <= 0.045

    def test_length_distribution_is_close(self, data):
        systems = [d["messages"][0]["content"] for d in data]
        users = [m["content"] for d in data for m in d["messages"]
                 if m.get("role") == "user"]
        assistants = [m["content"] for d in data for m in d["messages"]
                      if m.get("role") == "assistant"]
        lo, hi = profiles.REALISTIC_SYSTEM_CHARS
        assert all(lo <= len(s) <= hi for s in systems)
        assert 12 <= statistics.median(len(u) for u in users) <= 18
        assert 13 <= statistics.mean(len(u) for u in users) <= 18
        assert 42 <= statistics.median(len(a) for a in assistants) <= 55
        assert 48 <= statistics.mean(len(a) for a in assistants) <= 60

    def test_entity_density_is_consistent(self, data):
        for d in data:
            system = d["messages"][0]["content"]
            system_ids = set(_ID_PATTERN.findall(system))
            assert "订单金额" in system and _AMOUNT_PATTERN.search(system)
            assert system_ids
            for m in d["messages"][1:]:
                text = str(m.get("content") or "")
                if m.get("role") == "assistant":
                    assert set(_ID_PATTERN.findall(text)) <= system_ids
                    assert not _AMOUNT_PATTERN.search(text)

    def test_roles_alternate_and_no_adjacent_repetition(self, data):
        for d in data:
            roles = _roles(d)
            assert roles == (["system"] + ["user", "assistant"]
                             * ((len(roles) - 1) // 2))
            assistants = [m["content"] for m in d["messages"]
                          if m.get("role") == "assistant"]
            for a, b in zip(assistants, assistants[1:]):
                assert _jacobian(a, b) < 0.8

    def test_every_defect_type_can_be_injected(self):
        data = generate_dialogues(20, profiles.REALISTIC_BASE_SEED,
                                  profile=profiles.PROFILE_REALISTIC)
        records = generate(data, per_type=20, seed=20261113)
        skipped = [r for r in records if is_skipped(r)]
        assert not skipped
        assert {r["defect"]["type"] for r in records} == set(DEFECT_TYPES)


class TestCommittedRealisticFixtures:
    def test_base_matches_generator(self):
        data = generate_dialogues(
            profiles.REALISTIC_BASE_N, profiles.REALISTIC_BASE_SEED,
            profile=profiles.PROFILE_REALISTIC)
        assert _load("realistic_v1.json") == data

    def test_split_matches_generator(self):
        dev, test = generate_split(
            profiles.REALISTIC_DEV_N, profiles.REALISTIC_TEST_N,
            profiles.REALISTIC_SPLIT_SEED, profile=profiles.PROFILE_REALISTIC)
        assert _load("realistic_dev_v1.json") == dev
        assert _load("realistic_test_v1.json") == test
        assert {json.dumps(d, sort_keys=True, ensure_ascii=False) for d in dev}.isdisjoint(
            {json.dumps(d, sort_keys=True, ensure_ascii=False) for d in test})
