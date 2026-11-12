"""测试：条目级断点

为什么需要：LLM 逐条调用一旦中断，步骤级断点会整步重跑 → 重复计费。
这里以内容哈希为键，逐条记录，支持续跑时跳过已完成条目。
"""

import json

from agent.runtime.checkpoint import ItemCheckpoint


class TestCheckpointMemory:
    def test_new_key_not_done(self):
        cp = ItemCheckpoint()
        assert cp.is_done("k1") is False

    def test_mark_done(self):
        cp = ItemCheckpoint()
        cp.mark("k1", "accepted")
        assert cp.is_done("k1") is True

    def test_fallback_counts_as_done(self):
        cp = ItemCheckpoint()
        cp.mark("k1", "fallback")
        assert cp.is_done("k1") is True

    def test_failed_not_done(self):
        """失败条目必须重试，不能当作完成。"""
        cp = ItemCheckpoint()
        cp.mark("k1", "failed")
        assert cp.is_done("k1") is False

    def test_completed_count(self):
        cp = ItemCheckpoint()
        cp.mark("a", "accepted")
        cp.mark("b", "fallback")
        cp.mark("c", "failed")
        assert cp.completed == 2


class TestCheckpointPersistence:
    def test_written_as_jsonl(self, tmp_path):
        p = tmp_path / "state.jsonl"
        with ItemCheckpoint(p) as cp:
            cp.mark("k1", "accepted", tokens=12)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert rec["key"] == "k1" and rec["status"] == "accepted"
        assert rec["tokens"] == 12

    def test_resumes_from_existing_file(self, tmp_path):
        p = tmp_path / "state.jsonl"
        with ItemCheckpoint(p) as cp:
            cp.mark("k1", "accepted")
        # 新实例应能读到历史，避免重跑
        cp2 = ItemCheckpoint(p)
        try:
            assert cp2.is_done("k1") is True
        finally:
            cp2.close()

    def test_appends_not_truncates(self, tmp_path):
        p = tmp_path / "state.jsonl"
        with ItemCheckpoint(p) as cp:
            cp.mark("k1", "accepted")
        with ItemCheckpoint(p) as cp:
            cp.mark("k2", "accepted")
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_ignores_corrupted_line(self, tmp_path):
        p = tmp_path / "state.jsonl"
        p.write_text('{"key": "k1", "status": "accepted"}\nnot-json\n', encoding="utf-8")
        cp = ItemCheckpoint(p)
        try:
            assert cp.is_done("k1") is True
        finally:
            cp.close()
