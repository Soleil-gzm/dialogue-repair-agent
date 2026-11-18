"""运行环境 manifest：只溯源，不判等

与 baseline_lock 的分工（ADR D7）：
- `baseline_lock.json`：跨机器必须一致 → 不一致即结论作废
- `run_manifest.json`：这次结果在哪台机器、什么环境下跑出来的 → 换机不同属正常

因此 manifest 不入库、不参与 `--verify`；它只保证"结果有出处"。
"""

import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .canonical import sha256_file, stable_dump

# 优先记录锁定文件；两者都没有时显式留空，避免"看起来锁过"
LOCK_FILES = ("uv.lock", "requirements.lock")


def _deps_lock_sha(root: Path) -> Optional[Dict[str, str]]:
    for name in LOCK_FILES:
        p = root / name
        if p.exists():
            return {"file": name, "sha256": sha256_file(p)}
    return None


def build_manifest(
    *,
    root: Optional[Path] = None,
    profile: str = "standard",
    arms: Optional[Iterable[str]] = None,
    models: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(root) if root is not None else Path.cwd()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "executable": sys.executable,
        },
        "deps_lock": _deps_lock_sha(root),
        "profile": profile,
        "arms": list(arms or []),
        "models": models or {},
        "extra": extra or {},
    }


def write_manifest(manifest: Dict[str, Any], path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(stable_dump(manifest))
    return p
