"""条目级断点

为什么不用主工程的 StateTracker：它的粒度是"步骤完成"，而 LLM 逐条处理一旦中断，
整步重跑会重复计费。这里以对话内容哈希为键，逐条记录状态，支持续跑。
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

DONE_STATUSES = {"accepted", "fallback", "skipped"}


class ItemCheckpoint:
    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._path = Path(path) if path else None
        self._done: Dict[str, str] = {}
        self._records: Dict[str, Dict[str, Any]] = {}
        self._fh = None

        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                self._load()
            self._fh = open(self._path, "a", encoding="utf-8")

    def _load(self) -> None:
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key, status = rec.get("key"), rec.get("status")
            if key and status:
                self._done[key] = status
                self._records[key] = rec

    def is_done(self, key: str) -> bool:
        return self._done.get(key) in DONE_STATUSES

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """取已完成条目的完整记录（含额外字段）；未完成返回 None。

        批量评测续跑需要的不只是"做过没有"，还要还原该条目的产出（如是否检出），
        否则续跑后的分母与指标会与一次性跑完不一致。
        """
        return self._records.get(key) if self.is_done(key) else None

    def mark(self, key: str, status: str, **extra: Any) -> None:
        self._done[key] = status
        rec = {"key": key, "status": status, **extra}
        self._records[key] = rec
        if self._fh is not None:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def completed(self) -> int:
        return sum(1 for s in self._done.values() if s in DONE_STATUSES)
