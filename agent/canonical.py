"""规范化序列化与哈希（全项目唯一入口）

任何指纹、快照、缓存 key、对话主键都必须走这里，不允许各模块各自 json.dumps。
放在 agent 根层而不是 benchmark 下，是为了让 runtime / eval 也能复用，
同时避免 runtime 反向依赖 benchmark。
"""

import hashlib
import json
from pathlib import Path


def stable_dump(obj) -> bytes:
    """规范化 JSON：UTF-8、不转义中文、缩进 2、键排序、末尾换行。"""
    return (
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_bytes(stable_dump(obj))


def sha256_file(path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(obj, length: int = 16) -> str:
    """内容哈希：用于对话/请求主键，长度可截断。

    保持历史序列化格式（默认分隔符），确保已提交 fixture 的排序不变。
    """
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def dialogue_key(dialogue) -> str:
    """对话内容哈希：数据没有稳定 id，下标跨 run 不稳定。"""
    return stable_key(dialogue)


__all__ = [
    "stable_dump", "sha256_bytes", "sha256_obj", "sha256_file",
    "stable_key", "dialogue_key",
]
