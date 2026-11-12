"""规范化序列化与哈希

唯一目的：让"同一份数据"在任何机器、任何时间都得到同一个 sha256。
所有指纹、快照、产物哈希必须走这里，不允许各自 json.dumps。
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
