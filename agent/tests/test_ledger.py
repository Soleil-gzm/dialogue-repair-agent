"""测试：Token 账本

门禁：调用次数、token、成本三个维度都必须可复算；缓存命中成本必须为 0。
"""

import json

from agent.runtime.ledger import LedgerEntry, TokenLedger


def _entry(**kw) -> LedgerEntry:
    base = dict(run_id="r1", dialogue_key="k1", seq=1, stage="check",
                input_tokens=100, output_tokens=50)
    base.update(kw)
    return LedgerEntry(**base)


class TestRecord:
    def test_total_tokens(self):
        with TokenLedger() as L:
            L.record(_entry(input_tokens=100, output_tokens=50))
            L.record(_entry(seq=2, input_tokens=10, output_tokens=5))
        assert L.total_tokens() == 165

    def test_total_cost(self):
        with TokenLedger() as L:
            L.record(_entry(cost_cny=0.001))
            L.record(_entry(seq=2, cost_cny=0.002))
        assert L.total_cost() == 0.003

    def test_cached_entry_costs_zero(self):
        with TokenLedger() as L:
            e = L.record(_entry(cached=True, cost_cny=9.99))
        assert e.cost_cny == 0.0
        assert L.total_cost() == 0.0

    def test_failed_call_still_recorded(self):
        """失败调用必须入账，否则成本口径偏低。"""
        with TokenLedger() as L:
            L.record(_entry(ok=False, error_type="timeout"))
        assert len(L.entries) == 1
        assert L.summary()["error_rate"] == 1.0


class TestSummary:
    def test_by_stage_and_by_arm(self):
        with TokenLedger() as L:
            L.record(_entry(stage="check", arm="A3", input_tokens=100, output_tokens=0))
            L.record(_entry(seq=2, stage="rewrite", arm="A3", input_tokens=200, output_tokens=100))
            L.record(_entry(seq=3, stage="check", arm="A2", input_tokens=50, output_tokens=50))
        s = L.summary()
        assert s["by_stage"]["check"]["calls"] == 2
        assert s["by_stage"]["check"]["tokens"] == 200
        assert s["by_stage"]["rewrite"]["tokens"] == 300
        assert s["by_arm"]["A3"]["tokens"] == 400
        assert s["by_arm"]["A2"]["calls"] == 1

    def test_cache_hit_rate(self):
        with TokenLedger() as L:
            L.record(_entry(cached=True))
            L.record(_entry(seq=2, cached=False))
        assert L.summary()["cache_hit_rate"] == 0.5

    def test_retry_rate(self):
        with TokenLedger() as L:
            L.record(_entry(retries=2))
            L.record(_entry(seq=2, retries=0))
        assert L.summary()["retry_rate"] == 0.5

    def test_cost_per_ok_call(self):
        with TokenLedger() as L:
            L.record(_entry(cost_cny=0.002, ok=True))
            L.record(_entry(seq=2, cost_cny=0.0, ok=False, error_type="http_500"))
        assert L.summary()["cost_per_ok_call"] == 0.002

    def test_empty_ledger(self):
        with TokenLedger() as L:
            s = L.summary()
        assert s["n_calls"] == 0 and s["cache_hit_rate"] == 0.0


class TestPersistence:
    def test_appends_jsonl(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        with TokenLedger(p) as L:
            L.record(_entry())
            L.record(_entry(seq=2))
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["stage"] == "check"

    def test_does_not_truncate_existing(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        with TokenLedger(p) as L:
            L.record(_entry())
        with TokenLedger(p) as L:
            L.record(_entry(seq=2))
        assert len(p.read_text(encoding="utf-8").strip().split("\n")) == 2

    def test_entry_is_json_serializable(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        with TokenLedger(p) as L:
            L.record(_entry(http_status=200))
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert rec["http_status"] == 200 and rec["ok"] is True
