"""四元组指纹：⟨输入, 配置, 评估器, 适配⟩

依据《Benchmark专项设计方案》§1.1 与《数据格式适配与兼容规范》§4.2：
基线不是一个"跑过的结果"，而是一组可复算的指纹。

设计要点：
- 配置指纹先剥离易变键（timestamp/run_id 等），否则每次运行都不同，基线无法锁定。
- 输入指纹同时记录内容 sha256 与样本数，防止"同名不同内容"。
- 评估器/适配指纹只记录版本号与参数，不记录实现代码（代码由 git commit 覆盖）。
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .canonical import sha256_file, sha256_obj


# 这些键在配置中易变，不参与指纹计算
VOLATILE_KEYS = frozenset({
    "timestamp", "run_id", "ts", "generated_at", "created_at",
    "output_dir", "tag", "source_file",
})


def strip_volatile(obj: Any) -> Any:
    """递归剔除易变键，保留其余结构。"""
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(x) for x in obj]
    return obj


def _count_records(path: Path) -> int:
    """统计记录数：.jsonl 按行，.json 按数组长度。"""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    if path.suffix.lower() == ".jsonl":
        return sum(1 for line in text.split("\n") if line.strip())
    data = json.loads(text)
    return len(data) if isinstance(data, list) else 1


def _portable_path(p: Path, root: Optional[Path]) -> str:
    """优先记录相对于项目根的路径，保证 lock 跨机器可比。"""
    try:
        base = Path(root).resolve() if root is not None else Path.cwd().resolve()
        rel = p.resolve().relative_to(base)
        return rel.as_posix()
    except (ValueError, OSError):
        return str(p).replace("\\", "/")


def input_fingerprint(
    path,
    *,
    root=None,
    sample_keys: Optional[Iterable[str]] = None,
    n_sampled: Optional[int] = None,
    sample_seed: Optional[int] = None,
    strata: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """输入指纹：内容哈希 + 规模 + 抽样信息。

    path 记录为相对 root（默认 cwd）的路径，避免绝对路径导致跨机器不可比。
    """
    p = Path(path)
    fp: Dict[str, Any] = {
        "path": _portable_path(p, root),
        "sha256": sha256_file(p),
        "n_total": _count_records(p),
    }
    if sample_keys is not None:
        keys = sorted(sample_keys)
        fp["n_sampled"] = n_sampled if n_sampled is not None else len(keys)
        fp["sample_keys_sha256"] = sha256_obj(keys)
        fp["sample_seed"] = sample_seed
        fp["strata"] = list(strata or [])
    return fp


def config_fingerprint(config: Dict[str, Any]) -> str:
    """配置指纹：剥离易变键后的规范化哈希。"""
    return sha256_obj(strip_volatile(config))


def evaluator_fingerprint(
    *,
    rules_version: str,
    judge_prompt_version: str,
    judge_model: str,
    rubric_version: str,
    judge_temperature: float = 0.0,
) -> Dict[str, Any]:
    """评估器指纹：换标准就必须换指纹。"""
    body = {
        "rules_version": rules_version,
        "judge_prompt_version": judge_prompt_version,
        "judge_model": judge_model,
        "rubric_version": rubric_version,
        "judge_temperature": judge_temperature,
    }
    body["sha256"] = sha256_obj(body)
    return body


def adapter_fingerprint(
    *,
    adapter_version: str,
    canonical_schema_version: str,
    domain_profile: str,
    domain_profile_hash: Optional[str] = None,
    profile_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """适配指纹：证明两次运行用的是同一把尺子。

    若未显式给 domain_profile_hash，则由 profile_config 计算，禁止留占位符。
    """
    body = {
        "adapter_version": adapter_version,
        "canonical_schema_version": canonical_schema_version,
        "domain_profile": domain_profile,
    }
    body["domain_profile_hash"] = (
        domain_profile_hash
        if domain_profile_hash is not None
        else sha256_obj(profile_config or body)
    )
    body["sha256"] = sha256_obj(body)
    return body


def build_lock(
    *,
    input_fp: Dict[str, Any],
    evaluator_fp: Dict[str, Any],
    adapter_fp: Dict[str, Any],
    arms: Dict[str, Dict[str, Any]],
    code: Optional[Dict[str, Any]] = None,
    price_table: Optional[Dict[str, Any]] = None,
    profile: str = "standard",
) -> Dict[str, Any]:
    """组装 baseline_lock。arms 形如 {"A0": {"config_sha256":..., "prompt_version": null}}"""
    return {
        "lock_version": "1.0",
        "profile": profile,
        "code": code or {"git_commit": None, "dirty": None},
        "input": input_fp,
        "evaluator": evaluator_fp,
        "adapter": adapter_fp,
        "price_table": price_table or {"version": "unset", "unit": "CNY/1M tokens"},
        "arms": arms,
    }
