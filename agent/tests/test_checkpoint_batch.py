"""测试：断点续跑接入批量评测

验收标准有两条：
1. 续跑结果与一次跑完完全一致（指标不能因跳过而漂移）；
2. 已完成条目不再触发任何模型调用（零重复计费）。
"""

import json

import pytest

from agent.benchmark.agent_eval import A2, A3, compare_arms, run_arm
from agent.runtime.checkpoint import ItemCheckpoint
from agent.runtime.llm import LLMResponse


class _EchoA2Client:
    """A2 桩：原样回显输入对话。"""

    def __init__(self):
        self.calls = 0
        self.arm = None

    def set_arm(self, arm):
        self.arm = arm

    def chat(self, messages, **kwargs):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False),
                           input_tokens=1)


class _BrokenA2Client:
    """A2 桩：返回不可解析文本，模拟瞬时故障。"""

    def __init__(self):
        self.calls = 0

    def set_arm(self, arm):
        pass

    def chat(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(text="not json", input_tokens=1)


class _ForbiddenClient:
    """一旦被调用就让用例失败：用于证明续跑没有产生任何真实调用。"""

    def __init__(self):
        self.calls = 0

    def set_arm(self, arm):
        pass

    def chat(self, messages, **kwargs):
        self.calls += 1
        raise AssertionError("续跑不应调用模型")


@pytest.fixture
def dataset(mini_dialogues):
    from agent.benchmark.defects import generate

    return mini_dialogues, generate(mini_dialogues, per_type=2, seed=7)


class TestItemCheckpoint:
    def test_get_returns_extra_fields_after_reload(self, tmp_path):
        path = tmp_path / "ck.jsonl"
        cp = ItemCheckpoint(path)
        cp.mark("A2:defect:k1", "accepted",
                outcome={"detected": True, "parse_error": False})
        cp.close()

        reopened = ItemCheckpoint(path)
        rec = reopened.get("A2:defect:k1")
        assert rec["outcome"]["detected"] is True
        assert reopened.is_done("A2:defect:k1")
        reopened.close()

    def test_get_returns_none_for_unfinished_status(self, tmp_path):
        cp = ItemCheckpoint(tmp_path / "ck.jsonl")
        cp.mark("k", "failed", outcome={})
        assert cp.is_done("k") is False
        assert cp.get("k") is None
        cp.close()

    def test_mark_without_path_is_in_memory(self):
        cp = ItemCheckpoint()
        cp.mark("k", "accepted", outcome={"detected": False})
        assert cp.get("k")["outcome"] == {"detected": False}


class TestResumeSingleArm:
    def test_a2_resume_reuses_outcomes_without_calls(self, dataset, tmp_path):
        base, defects = dataset
        cp_path = tmp_path / "ck.jsonl"

        first_client = _EchoA2Client()
        first = run_arm(A2, base, defects, client=first_client,
                        checkpoint=ItemCheckpoint(cp_path))
        assert first.resumed == 0
        assert first.calls == 0  # 无账本时 calls 由账本统计，这里只证明跑过
        assert first_client.calls == len(defects) + len(base)

        forbidden = _ForbiddenClient()
        second = run_arm(A2, base, defects, client=forbidden,
                         checkpoint=ItemCheckpoint(cp_path))
        assert forbidden.calls == 0, "已完成条目不得再次调用模型"
        assert second.resumed == len(defects) + len(base)
        assert second.per_type == first.per_type
        assert second.clean_flagged == first.clean_flagged
        assert second.failed == first.failed == 0

    def test_rule_arm_resume_matches_metrics(self, dataset, tmp_path):
        base, defects = dataset
        cp_path = tmp_path / "ck.jsonl"

        first = run_arm(A3, base, defects, checkpoint=ItemCheckpoint(cp_path))
        second = run_arm(A3, base, defects, checkpoint=ItemCheckpoint(cp_path))

        assert second.resumed == len(defects) + len(base)
        assert second.per_type == first.per_type
        assert second.clean_flagged == first.clean_flagged
        assert second.clean_total == first.clean_total

    def test_different_arm_does_not_reuse(self, dataset, tmp_path):
        base, defects = dataset
        cp_path = tmp_path / "ck.jsonl"

        run_arm(A2, base, defects, client=_EchoA2Client(),
                checkpoint=ItemCheckpoint(cp_path))
        other = run_arm(A3, base, defects, checkpoint=ItemCheckpoint(cp_path))
        assert other.resumed == 0, "断点键必须含臂名，换臂不能误跳过"

    def test_failed_items_are_retried(self, dataset, tmp_path):
        base, defects = dataset
        cp_path = tmp_path / "ck.jsonl"

        broken = _BrokenA2Client()
        first = run_arm(A2, base, defects, client=broken,
                        checkpoint=ItemCheckpoint(cp_path))
        assert first.failed == len(defects) + len(base)

        echo = _EchoA2Client()
        second = run_arm(A2, base, defects, client=echo,
                         checkpoint=ItemCheckpoint(cp_path))
        assert second.resumed == 0, "解析失败未完成，必须重试"
        assert echo.calls == len(defects) + len(base)
        assert second.failed == 0

    def test_resumed_samples_keep_review_material(self, dataset, tmp_path):
        base, defects = dataset
        cp_path = tmp_path / "ck.jsonl"

        first = run_arm(A3, base, defects, checkpoint=ItemCheckpoint(cp_path))
        second = run_arm(A3, base, defects, checkpoint=ItemCheckpoint(cp_path))
        # A3 在无改写时不产生 repaired 样本；此断言保证样本列表口径一致
        assert len(second.samples) == len(first.samples)


class TestResumeCompare:
    def test_compare_resume_makes_no_calls(self, dataset, tmp_path):
        base, defects = dataset
        cp_path = tmp_path / "ck.jsonl"

        echo = _EchoA2Client()
        first = compare_arms(A3, A2, base, defects, client=echo,
                             metric="clean", checkpoint=ItemCheckpoint(cp_path))
        assert first["resumed"] == 0
        assert echo.calls == len(base)

        forbidden = _ForbiddenClient()
        second = compare_arms(A3, A2, base, defects, client=forbidden,
                              metric="clean", checkpoint=ItemCheckpoint(cp_path))
        assert forbidden.calls == 0
        assert second["resumed"] == len(first["arms"]) * len(base)
        assert second["win_rate"] == first["win_rate"]
        assert second["mcnemar"] == first["mcnemar"]
