"""测试：决策链 Trace"""

import json

from agent.runtime.trace import TraceWriter


class TestTraceMemory:
    def test_seq_increments(self):
        t = TraceWriter()
        t.write({"event": "a"})
        t.write({"event": "b"})
        assert [r["seq"] for r in t.records] == [1, 2]
        assert t.count == 2

    def test_adds_timestamp(self):
        t = TraceWriter()
        rec = t.write({"event": "a"})
        assert "t_ms" in rec and rec["t_ms"] >= 0

    def test_keeps_payload(self):
        t = TraceWriter()
        t.write({"event": "decide", "tool": "x", "reason": "r"})
        assert t.records[0]["tool"] == "x"


class TestTraceFile:
    def test_writes_jsonl(self, tmp_path):
        p = tmp_path / "trace.jsonl"
        with TraceWriter(p) as t:
            t.write({"event": "a"})
            t.write({"event": "b"})
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert [json.loads(l)["event"] for l in lines] == ["a", "b"]

    def test_flushes_immediately(self, tmp_path):
        """进程被杀也要留下已发生的决策：写完即可读到。"""
        p = tmp_path / "trace.jsonl"
        t = TraceWriter(p)
        try:
            t.write({"event": "a"})
            assert len(p.read_text(encoding="utf-8").strip().split("\n")) == 1
        finally:
            t.close()

    def test_creates_parent_dir(self, tmp_path):
        p = tmp_path / "a" / "b" / "trace.jsonl"
        with TraceWriter(p) as t:
            t.write({"event": "a"})
        assert p.exists()

    def test_close_idempotent(self, tmp_path):
        t = TraceWriter(tmp_path / "t.jsonl")
        t.close()
        t.close()

    def test_no_path_does_not_write(self, tmp_path):
        t = TraceWriter()
        t.write({"event": "a"})
        assert list(tmp_path.iterdir()) == []
