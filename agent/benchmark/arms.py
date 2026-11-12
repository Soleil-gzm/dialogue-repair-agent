"""对照臂执行器（T1 只实现 A0 / A1）

- A0：不处理（恒等臂），作为下界基线
- A1：复用既有 05_augment 规则增强，作为"现有能力"基线

A2/A3/A4 依赖 LLM 与 Agent 循环，留待 M3。
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


# A1 默认增强器组合：全部为词法/语序类，不含模型类（asr_noise），保证无重依赖
DEFAULT_A1_AUGMENTERS = {
    "insert_filler": {"weight": 1.0},
    "synonym_replace": {"weight": 1.0},
    "reorder": {"weight": 1.0},
}


def _build_a1_config(
    input_json: Path,
    workdir: Path,
    *,
    seed: int,
    num_variants: int,
    augmenters: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task_name": "bench_a1",
        "paths": {
            "intermediate": str(workdir / "intermediate"),
            "output": str(workdir / "output"),
        },
        "steps": {
            "05_augment": {
                "input_file": str(input_json),
                "output_dir": "{task_dir}/output_augmented_data",
                "seed": seed,
                "num_variants": num_variants,
                "target_roles": ["assistant"],
                "only_loss_true": True,
                "strategy": "single",
                "min_steps": 1,
                "max_steps": 1,
                "augmenters": augmenters,
            }
        },
        "logging": {"level": "ERROR"},
    }


def run_a1(
    dialogues: List[dict],
    workdir,
    *,
    seed: int = 42,
    num_variants: int = 1,
    augmenters: Optional[Dict[str, Any]] = None,
) -> ArmResult:
    """A1 臂：调用既有 AugmentStep，返回其 combined 产物指纹。

    注意：AugmentStep 会写出带时间戳的 run_id 目录，因此产物路径不参与指纹，
    只对数据内容取 sha256。
    """
    import json

    from pipeline.core.config_manager import ConfigManager
    from pipeline.core.path_resolver import PathResolver
    from pipeline.core.state_tracker import StateTracker
    from pipeline.steps.augment import AugmentStep

    workdir = Path(workdir).resolve()  # 必须绝对路径：PathResolver 会把相对路径拼到 project_root
    workdir.mkdir(parents=True, exist_ok=True)
    input_json = workdir / "a1_input.json"
    input_json.write_text(json.dumps(dialogues, ensure_ascii=False), encoding="utf-8")

    cfg = _build_a1_config(
        input_json, workdir,
        seed=seed, num_variants=num_variants,
        augmenters=augmenters or DEFAULT_A1_AUGMENTERS,
    )
    pr = PathResolver(cfg)
    step = AugmentStep(ConfigManager(cfg), pr, StateTracker(pr.task_dir))

    ok = step.run()
    if not ok:
        raise RuntimeError("A1 臂执行失败（AugmentStep.run 返回 False）")

    out_root = pr.resolve("{task_dir}/output_augmented_data")
    run_dirs = sorted([d for d in out_root.iterdir() if d.is_dir()],
                      key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise RuntimeError(f"A1 未产出目录: {out_root}")
    latest = run_dirs[-1]

    combined = sorted(latest.glob("combined_augmented_*.json"))[0]
    data = json.loads(combined.read_text(encoding="utf-8"))

    return ArmResult(
        arm="A1", n_in=len(dialogues), n_out=len(data),
        artifact_sha256=sha256_obj(data), artifact_path=combined,
        meta={"method": "existing_augment", "seed": seed,
              "num_variants": num_variants,
              "augmenters": sorted((augmenters or DEFAULT_A1_AUGMENTERS).keys())},
    )
