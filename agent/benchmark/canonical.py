"""规范化序列化与哈希（兼容导出）

唯一实现在 `agent.canonical`；这里保留模块路径，避免既有 import 断裂。
所有指纹、快照、产物哈希统一走那一份实现。
"""

from ..canonical import (
    dialogue_key,
    sha256_bytes,
    sha256_file,
    sha256_obj,
    stable_dump,
    stable_key,
)

__all__ = [
    "stable_dump", "sha256_bytes", "sha256_obj", "sha256_file",
    "stable_key", "dialogue_key",
]
