"""测试：对照臂 A0

关键门禁（T1）：
- A0 的产物指纹必须与冻结快照一致 —— 证明"基线"与"已固化的期望值"是同一个东西

A1（上游加噪增强）已于 2026-11-19 移出主消融链，相关测试一并移除；
原因与归档见 docs/notes/2026-11-19-A1臂移除记录.md。
"""

from agent.benchmark.arms import run_a0
from agent.benchmark.canonical import sha256_obj
from agent.benchmark.fixtures import generate_dialogues
from agent.tests.conftest import assert_snapshot


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
