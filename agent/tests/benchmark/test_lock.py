"""单元测试：agent.benchmark.lock（不可变性是核心）"""

import json

import pytest

from agent.benchmark.lock import (
    LockViolation, write_lock, read_lock, verify_lock,
)


def _lock(seed: int = 1) -> dict:
    return {
        "lock_version": "1.0",
        "generated_at": "2026-11-12T10:00:00",
        "input": {"sha256": f"in{seed}", "n_total": 3},
        "arms": {"A0": {"config_sha256": "c"}},
    }


class TestWriteLock:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(), p)
        assert p.exists()

    def test_idempotent_when_identical(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(), p)
        write_lock(_lock(), p)  # 不应抛
        assert read_lock(p)["input"]["sha256"] == "in1"

    def test_refuses_overwrite_on_change(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(1), p)
        with pytest.raises(LockViolation):
            write_lock(_lock(2), p)

    def test_force_requires_reason(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(1), p)
        with pytest.raises(ValueError):
            write_lock(_lock(2), p, force=True, reason="")

    def test_force_with_reason_overwrites_and_logs(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(1), p)
        write_lock(_lock(2), p, force=True, reason="更换评估器版本")
        assert read_lock(p)["input"]["sha256"] == "in2"
        log = tmp_path / "lock_changelog.md"
        assert log.exists() and "更换评估器版本" in log.read_text(encoding="utf-8")

    def test_generated_at_change_is_not_a_violation(self, tmp_path):
        """generated_at 是唯一允许变动的字段；仅它变化时视为无操作，不改写文件。"""
        p = tmp_path / "baseline_lock.json"
        a = _lock(1); a["generated_at"] = "2026-11-12T10:00:00"
        b = _lock(1); b["generated_at"] = "2026-11-13T11:00:00"
        write_lock(a, p)
        write_lock(b, p)  # 不应抛
        # 幂等：内容等价时不重写，保留首次生成时间
        assert read_lock(p)["generated_at"] == "2026-11-12T10:00:00"

    def test_written_file_is_canonical_json(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(), p)
        raw = p.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        assert json.loads(raw)["lock_version"] == "1.0"


class TestVerifyLock:
    def test_identical_returns_empty(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(), p)
        assert verify_lock(p, _lock()) == []

    def test_detects_input_change(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(1), p)
        diffs = verify_lock(p, _lock(2))
        assert any("input" in d for d in diffs)

    def test_ignores_generated_at(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        write_lock(_lock(), p)
        current = _lock(); current["generated_at"] = "2099-01-01T00:00:00"
        assert verify_lock(p, current) == []

    def test_ignores_code_dirty_but_detects_commit_change(self, tmp_path):
        p = tmp_path / "baseline_lock.json"
        a = _lock(); a["code"] = {"git_commit": "abc", "dirty": False}
        write_lock(a, p)

        # dirty 翻转不算基线变化
        b = _lock(); b["code"] = {"git_commit": "abc", "dirty": True}
        assert verify_lock(p, b) == []

        # commit 变化必须报出
        c = _lock(); c["code"] = {"git_commit": "def", "dirty": True}
        assert any("code" in d for d in verify_lock(p, c))
