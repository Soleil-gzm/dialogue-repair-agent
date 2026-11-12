"""决策链 Trace

每一步都逐行 flush，进程被杀也能留下已发生的决策——这是"决策可解释"的物理基础。
不写 prompt 全文、不写对话内容，只写结构化摘要，避免敏感信息落盘。
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TraceWriter:
    """JSONL 逐行写入；未给路径时只保留在内存（供测试断言）。"""

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._path = Path(path) if path else None
        self._fh = None
        self.records: List[Dict[str, Any]] = []
        self._seq = 0
        self._t0 = time.monotonic()

        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """追加一条记录，补 seq 与相对耗时。返回写入后的记录。"""
        self._seq += 1
        item = {
            "seq": self._seq,
            "t_ms": int((time.monotonic() - self._t0) * 1000),
            **record,
        }
        self.records.append(item)
        if self._fh is not None:
            self._fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._fh.flush()  # 决策链必须实时可见
        return item

    @property
    def count(self) -> int:
        return self._seq

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
