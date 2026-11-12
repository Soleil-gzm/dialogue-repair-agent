"""Token 账本（只计量，不管控）

每一次 LLM 调用都留一行，append-only、写完即 flush。
失败调用、缓存命中同样入账——否则成本口径偏低，分母也不真实。

定价不放代码里：只记 price_table_version 与折算后的 cost_cny，
日后调价可重算而不污染历史记录。
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class LedgerEntry:
    run_id: str
    dialogue_key: str
    seq: int
    stage: str                     # check / rewrite / evaluate / decide
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    arm: str = "A3"
    provider: str = "deepseek"
    model: str = ""
    prompt_version: str = ""
    latency_ms: int = 0
    retries: int = 0
    http_status: Optional[int] = None
    cached: bool = False
    cost_cny: float = 0.0
    price_table_version: str = "unset"
    request_hash: str = ""
    ok: bool = True
    error_type: Optional[str] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenLedger:
    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._path = Path(path) if path else None
        self.entries: List[LedgerEntry] = []
        self._fh = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "a", encoding="utf-8")

    def record(self, entry: LedgerEntry) -> LedgerEntry:
        # 缓存命中不计费：保证"重复运行的真实成本"可算
        if entry.cached:
            entry.cost_cny = 0.0
        self.entries.append(entry)
        if self._fh is not None:
            self._fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            self._fh.flush()
        return entry

    # ------------------------------------------------------------------ 汇总
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.entries)

    def total_cost(self) -> float:
        return round(sum(e.cost_cny for e in self.entries), 8)

    def _group(self, key: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for e in self.entries:
            k = getattr(e, key)
            g = out.setdefault(k, {"calls": 0, "tokens": 0, "cost_cny": 0.0})
            g["calls"] += 1
            g["tokens"] += e.total_tokens
            g["cost_cny"] = round(g["cost_cny"] + e.cost_cny, 8)
        return out

    def summary(self) -> Dict[str, Any]:
        n = len(self.entries)
        accepted = sum(1 for e in self.entries if e.ok and not e.error_type)
        return {
            "n_calls": n,
            "total_tokens": self.total_tokens(),
            "total_cost_cny": self.total_cost(),
            "by_stage": self._group("stage"),
            "by_arm": self._group("arm"),
            "cache_hit_rate": round(
                sum(1 for e in self.entries if e.cached) / n, 4) if n else 0.0,
            "error_rate": round(
                sum(1 for e in self.entries if not e.ok) / n, 4) if n else 0.0,
            "retry_rate": round(
                sum(1 for e in self.entries if e.retries > 0) / n, 4) if n else 0.0,
            "ok_calls": accepted,
            "cost_per_ok_call": (
                round(self.total_cost() / accepted, 8) if accepted else 0.0
            ),
        }

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
