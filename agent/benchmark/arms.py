"""对照臂执行器（run.py 侧只实现 A0）

- A0：不处理（恒等臂），作为下界基线

A2/A3/A4 依赖 LLM 与 Agent 循环，统一走 agent_eval.py。

历史上的 A1（上游 05_augment 加噪增强）已于 2026-11-19 移出主消融链：
它做的是加噪而非语义修复，输出变体不可配对，且脱离上游环境不可复现。
旧指纹归档在 baseline/deprecated/a1_2026-11-19.json，原因见
docs/notes/2026-11-19-A1臂移除记录.md。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .canonical import sha256_obj


@dataclass
class ArmResult:
    arm: str
    n_in: int
    n_out: int
    artifact_sha256: str
    artifact_path: Optional[Path] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def run_a0(dialogues: List[dict]) -> ArmResult:
    """恒等臂：不对数据做任何处理。用于确认下界与快照一致性。"""
    digest = sha256_obj(dialogues)
    return ArmResult(
        arm="A0", n_in=len(dialogues), n_out=len(dialogues),
        artifact_sha256=digest, meta={"method": "identity"},
    )
