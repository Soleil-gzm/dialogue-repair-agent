"""Benchmark harness（T1 基线冻结阶段）

与主工程解耦：不 import pipeline_agent，仅在 A1 臂中复用既有 05_augment 步骤。
"""

from .canonical import stable_dump, sha256_bytes, sha256_obj, sha256_file
from .fingerprint import (
    input_fingerprint,
    config_fingerprint,
    evaluator_fingerprint,
    adapter_fingerprint,
    build_lock,
)
from .lock import LockViolation, write_lock, read_lock, verify_lock

__all__ = [
    "stable_dump",
    "sha256_bytes",
    "sha256_obj",
    "sha256_file",
    "input_fingerprint",
    "config_fingerprint",
    "evaluator_fingerprint",
    "adapter_fingerprint",
    "build_lock",
    "LockViolation",
    "write_lock",
    "read_lock",
    "verify_lock",
]
