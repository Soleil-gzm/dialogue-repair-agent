"""测试：公开轨 fixture 的 dev/test 分层切分

关键不是"切出来了"，而是"切得可复现、无重叠、单例层不丢测试覆盖"。
"""

import json
import random
from pathlib import Path

import pytest

from agent.benchmark.fixtures import (
    DEV_N, SPLIT_SEED, TEST_N, dialogue_key, generate_dialogues,
    generate_split, make_dialogue, split_dialogues,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "public"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _with_scene(dialogue: dict, scene: str) -> dict:
    for msg in dialogue["messages"]:
        if msg.get("role") == "system":
            head, tail = str(msg["content"]).split("## 场景标签", 1)
            rest = tail.split("\n", 2)[2]  # 丢掉原场景行，保留后续段落
            msg["content"] = head + "## 场景标签\n" + scene + "\n" + rest
            break
    return dialogue


class TestSplit:
    def test_counts_and_no_overlap(self):
        dev, test = generate_split()
        assert len(dev) == DEV_N
        assert len(test) == TEST_N
        assert {dialogue_key(d) for d in dev}.isdisjoint(
            {dialogue_key(d) for d in test})

    def test_reproducible(self):
        assert generate_split() == generate_split()

    def test_order_independent(self):
        pool = generate_dialogues(DEV_N + TEST_N, SPLIT_SEED)
        dev, test = split_dialogues(pool, DEV_N, TEST_N)
        rev_dev, rev_test = split_dialogues(list(reversed(pool)), DEV_N, TEST_N)
        assert dev == rev_dev and test == rev_test

    def test_requires_full_partition(self):
        pool = generate_dialogues(5, 3)
        with pytest.raises(ValueError):
            split_dialogues(pool, 2, 2)

    def test_singleton_stratum_goes_to_test(self):
        rng = random.Random(11)
        pool = [_with_scene(make_dialogue(i, 4, rng), "S1_咨询") for i in range(3)]
        odd = make_dialogue(9, 5, rng)  # 独有场景 + 轮次桶，形成单例层
        pool.append(odd)

        dev, test = split_dialogues(pool, dev_n=2, test_n=2)
        assert dialogue_key(odd) in {dialogue_key(d) for d in test}
        assert len(dev) == 2 and len(test) == 2

    def test_committed_files_match_generator(self):
        dev, test = generate_split()
        assert _load("dev_v1.json") == dev
        assert _load("test_v1.json") == test
