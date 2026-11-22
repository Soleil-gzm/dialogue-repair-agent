"""baseline_lock 的写入与校验

核心规则（《Benchmark专项设计方案》§1.1）：
**baseline_lock.json 一经生成即不可修改。**
指纹变化必须新建 lock 并说明原因，禁止原地覆盖——否则"基线偷偷漂移"。

`generated_at` 与 `code` 仅记录生成时刻与代码出处，不参与校验：
前者是时间戳，后者会因提交 lock 本身而变化。
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from .canonical import stable_dump

# 不参与跨版本比对的键
# code 段只做溯源、不做判等：提交 lock 本身就会改变 HEAD，若比较 git_commit，
# `--verify` 在锁入库后必然失败。真正该判等的是四元组指纹。
_IGNORED_ON_VERIFY = frozenset({"generated_at", "code"})

# 嵌套键中不参与比对的项（保留给锁内其他嵌套结构）
_IGNORED_NESTED: set = set()


class LockViolation(RuntimeError):
    """试图覆盖已存在的 baseline_lock，或校验失败。"""


def read_lock(path) -> Dict[str, Any]:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise LockViolation(
            f"baseline_lock 不存在: {p}；请先用 run --write-lock 生成，而不是直接校验。"
        ) from e


def _comparable(lock: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in lock.items() if k not in _IGNORED_ON_VERIFY}
    for parent, child in _IGNORED_NESTED:
        if isinstance(out.get(parent), dict):
            out[parent] = {k: v for k, v in out[parent].items() if k != child}
    return out


def write_lock(
    lock: Dict[str, Any],
    path,
    *,
    force: bool = False,
    reason: str = "",
    allow_dirty: bool = False,
) -> Path:
    """写入 lock。已存在且内容不同时默认拒绝。

    Args:
        force: 显式覆盖开关
        reason: force=True 时必填，用于写入 lock_changelog
        allow_dirty: 允许在工作区有未提交改动时生成锁（默认拒绝）
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    dirty = (lock.get("code") or {}).get("dirty")
    if dirty is True and not allow_dirty:
        raise LockViolation(
            "工作区有未提交改动，拒绝生成 baseline_lock："
            "git_commit 无法完整描述产生指纹的代码。请先提交，或显式 --allow-dirty 仅作临时调试。"
        )

    changelog_reason = ""
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
        changelog_reason = reason

    # 先落 lock，再记 changelog：写 lock 失败时不能留下"已覆盖"的假记录
    p.write_bytes(stable_dump(lock))
    if changelog_reason:
        _append_changelog(p, changelog_reason)
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
