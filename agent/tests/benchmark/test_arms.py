"""测试：对照臂 A0 / A1

关键门禁（T1）：
- A0 的产物指纹必须与冻结快照一致 —— 证明"基线"与"已固化的期望值"是同一个东西
- A1 必须可复现 —— 同 seed 两次运行的产物指纹相同
"""

import importlib.util
import json

import pytest

from agent.benchmark.arms import run_a0, run_a1, DEFAULT_A1_AUGMENTERS
from agent.benchmark.canonical import sha256_obj
from agent.benchmark.fixtures import generate_dialogues
from agent.tests.conftest import assert_snapshot

HAS_PIPELINE = importlib.util.find_spec("pipeline") is not None


class TestArmA0:
    def test_identity_preserves_size(self, mini_dialogues):
        r = run_a0(mini_dialogues)
        assert r.arm == "A0"
        assert r.n_in == r.n_out == len(mini_dialogues)

    def test_fingerprint_is_deterministic(self, mini_dialogues):
        assert run_a0(mini_dialogues).artifact_sha256 == run_a0(mini_dialogues).artifact_sha256

    def test_matches_frozen_fingerprint(self, update_snapshots):
        """T1 门禁：A0 对固定输入的产物指纹 == 冻结值。"""
        d = generate_dialogues(1, seed=7)[0]
        assert_snapshot("a0_single_dialogue", d, update_snapshots)
        # A0 是恒等臂，单条输入的产物指纹应等于该对象自身指纹
        assert run_a0([d]).artifact_sha256 == sha256_obj([d])


class TestArmA1:
    # A1 复用上游数据工程的增强步骤，脱离上游环境不可运行
    pytestmark = pytest.mark.skipif(
        not HAS_PIPELINE, reason="A1 臂需要上游数据工程环境（pipeline 包）"
    )

    def test_runs_and_expands(self, mini_dialogues, tmp_path):
        r = run_a1(mini_dialogues, tmp_path / "a1", seed=42, num_variants=1)
        assert r.arm == "A1"
        assert r.n_in == len(mini_dialogues)
        assert r.n_out >= len(mini_dialogues)  # 至少保留原始

    def test_original_data_preserved(self, mini_dialogues, tmp_path):
        r = run_a1(mini_dialogues, tmp_path / "a1", seed=42)
        data = json.loads(r.artifact_path.read_text(encoding="utf-8"))
        # 前 n 条应为原始数据（AugmentStep 先放 original 再放 variants）
        assert data[: len(mini_dialogues)] == mini_dialogues

    def test_reproducible_same_seed(self, mini_dialogues, tmp_path):
        a = run_a1(mini_dialogues, tmp_path / "r1", seed=42)
        b = run_a1(mini_dialogues, tmp_path / "r2", seed=42)
        assert a.artifact_sha256 == b.artifact_sha256, "A1 同 seed 结果不可复现"

    def test_no_model_augmenter_by_default(self):
        from pipeline.augmenters.categories import requires_model
        for name in DEFAULT_A1_AUGMENTERS:
            assert not requires_model(name), f"{name} 需要模型，不应作为默认 A1 增强器"
