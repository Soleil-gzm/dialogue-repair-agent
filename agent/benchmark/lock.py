"""baseline_lock 的写入与校验

核心规则（《Benchmark专项设计方案》§1.1）：
**baseline_lock.json 一经生成即不可修改。**
指纹变化必须新建 lock 并说明原因，禁止原地覆盖——否则"基线偷偷漂移"。

`generated_at` 是唯一允许变动的字段（仅记录生成时刻），不参与校验。
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from .canonical import stable_dump

# 不参与跨版本比对的键
_IGNORED_ON_VERIFY = frozenset({"generated_at"})

# 嵌套键中不参与比对的项：dirty 会随未提交文件增删而翻转，不代表基线变化
_IGNORED_NESTED = {("code", "dirty")}


class LockViolation(RuntimeError):
    """试图覆盖已存在的 baseline_lock，或校验失败。"""


def read_lock(path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _comparable(lock: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in lock.items() if k not in _IGNORED_ON_VERIFY}
    for parent, child in _IGNORED_NESTED:
        if isinstance(out.get(parent), dict):
            out[parent] = {k: v for k, v in out[parent].items() if k != child}
    return out


def write_lock(lock: Dict[str, Any], path, *, force: bool = False, reason: str = "") -> Path:
    """写入 lock。已存在且内容不同时默认拒绝。

    Args:
        force: 显式覆盖开关
        reason: force=True 时必填，用于写入 lock_changelog
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        existing = read_lock(p)
        if _comparable(existing) == _comparable(lock):
            return p  # 幂等：完全一致则不动
        if not force:
            raise LockViolation(
                f"基线已存在且内容不同，拒绝覆盖: {p}。"
                "如确需变更，请新建 lock 文件或显式 force 并说明原因。"
            )
        if not reason.strip():
            raise ValueError("force 覆盖 baseline_lock 必须提供 reason（将写入 lock_changelog）")
        _append_changelog(p, reason)

    p.write_bytes(stable_dump(lock))
    return p


def _append_changelog(lock_path: Path, reason: str) -> None:
    log = lock_path.parent / "lock_changelog.md"
    from datetime import datetime

    line = f"- {datetime.now().isoformat(timespec='seconds')} 覆盖 {lock_path.name} 原因：{reason}\n"
    if not log.exists():
        log.write_text("# 基线指纹变更记录\n\n" + line, encoding="utf-8")
    else:
        with open(log, "a", encoding="utf-8") as f:
            f.write(line)


def verify_lock(lock_path, current: Dict[str, Any]) -> List[str]:
    """比对当前计算出的指纹与已锁定指纹，返回差异描述列表（空表示一致）。"""
    locked = _comparable(read_lock(lock_path))
    now = _comparable(current)

    diffs: List[str] = []
    for key in sorted(set(locked) | set(now)):
        a, b = locked.get(key), now.get(key)
        if a != b:
            diffs.append(f"{key}: locked={a!r} current={b!r}")
    return diffs
