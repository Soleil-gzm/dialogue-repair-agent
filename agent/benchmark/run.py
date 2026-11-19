"""T1 基线冻结 CLI

用法（在仓库根目录执行）：
    python -m agent.benchmark.run --input agent/fixtures/public/base_v1.json --arms A0
    python -m agent.benchmark.run --input ... --verify   # 只校验，不重写 lock

本 CLI 只负责 A0（离线恒等臂）。A1 已于 2026-11-19 移出主消融链，
A2/A3/A3L/A3R/A4 统一走 agent_eval.py。

产出：
    {lock_dir}/baseline_lock.json   四元组指纹（只读，入库）
    {out}/report/arm_results.json   各臂产物指纹与规模（不入库）
    {out}/report/run_manifest.json  环境溯源（机器/依赖/模型，不入库、不判等）
"""

import argparse
import json
import os
import sys
from pathlib import Path

from agent.eval.judge import JUDGE_VERSION, RUBRIC_VERSION
from agent.eval.rules import RULES_VERSION
from agent.runtime.state import dialogue_key

from .arms import run_a0
from .fingerprint import (
    input_fingerprint,
    evaluator_fingerprint,
    adapter_fingerprint,
    build_lock,
)
from .lock import LockViolation, write_lock, read_lock, verify_lock
from .manifest import build_manifest, write_manifest


DEFAULT_EVALUATOR = dict(
    rules_version=RULES_VERSION,
    judge_prompt_version=JUDGE_VERSION,
    judge_model="none",  # A0 是离线恒等臂，不调任何 LLM
    rubric_version=RUBRIC_VERSION,
    judge_temperature=0.0,
)

DEFAULT_ADAPTER = dict(
    adapter_version="1.0.0",
    canonical_schema_version="1.0",
    domain_profile="public_after_sales",
)

DEFAULT_PRICE_TABLE = {"version": "deepseek_2026-11", "unit": "CNY/1M tokens"}

# 域配置：变更它即等于换域，必须升 major 并重建 lock
PUBLIC_DOMAIN_PROFILE = {
    "domain": "public_after_sales",
    "schema_version": "1.0",
    "tag_names": [
        "角色", "客户编号", "订单编号", "产品名称", "场景标签",
        "客服电话", "联系时间", "订单金额",
    ],
    "scenarios": ["S1_咨询", "S2_催办", "S3_退换", "S4_投诉", "S5_回访"],
}


def _load(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in text.split("\n") if l.strip()]
    return json.loads(text)


def _code_fingerprint() -> dict:
    """记录代码版本；git 不可用时降级为 None，不阻断。"""
    import subprocess

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, timeout=5)
        return {
            "git_commit": head.stdout.strip() or None,
            "dirty": bool(status.stdout.strip()),
        }
    except Exception:
        return {"git_commit": None, "dirty": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.run")
    ap.add_argument("--input", required=True, help="输入对话 JSON/JSONL")
    ap.add_argument("--arms", default="A0", help="本 CLI 只支持 A0")
    ap.add_argument("--out", default="agent/benchmark/output")
    ap.add_argument("--lock-dir", default="agent/benchmark/baseline",
                    help="基线目录（入库）；--out 为运行产物（不入库）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--verify", action="store_true", help="只校验已有 lock（要求臂集合完全一致）")
    ap.add_argument("--verify-arm", default=None,
                    help="只校验单个臂的产物是否复现（用于可独立运行的环境，如仅 A0）")
    ap.add_argument("--force", action="store_true", help="覆盖 lock（需 --reason）")
    ap.add_argument("--reason", default="")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="允许在工作区有未提交改动时生成锁（仅临时调试，入库锁必须 clean）")
    args = ap.parse_args(argv)

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"[ERROR] 输入不存在: {in_path}", file=sys.stderr)
        return 2

    dialogues = _load(in_path)
    arms_wanted = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    unsupported = [a for a in arms_wanted if a != "A0"]
    if unsupported:
        print(
            f"[ERROR] run.py 只支持 A0；其他臂请使用 agent_eval.py: {unsupported}",
            file=sys.stderr,
        )
        return 2

    results = {"A0": run_a0(dialogues)}

    arm_fps = {}
    for name, r in results.items():
        from .fingerprint import config_fingerprint
        arm_fps[name] = {
            "config_sha256": config_fingerprint({"method": r.arm}),
            "prompt_version": None,
            # 产物指纹：同输入同配置必然复现，可在无 LLM 环境中自证基线未漂移
            "artifact_sha256": r.artifact_sha256,
        }

    lock = build_lock(
        input_fp=input_fingerprint(
            in_path,
            root=Path.cwd(),
            # 当前不做抽样，但键集合与种子必须入锁：否则"换了一批样本"无法被发现
            sample_keys=[dialogue_key(d) for d in dialogues],
            n_sampled=len(dialogues),
            sample_seed=args.seed,
            strata=[],
        ),
        evaluator_fp=evaluator_fingerprint(**DEFAULT_EVALUATOR),
        adapter_fp=adapter_fingerprint(
            **DEFAULT_ADAPTER, profile_config=PUBLIC_DOMAIN_PROFILE
        ),
        arms=arm_fps,
        price_table=DEFAULT_PRICE_TABLE,
        code=_code_fingerprint(),
        profile=args.profile,
    )

    report_dir = Path(args.out) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(args.lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "baseline_lock.json"

    if args.verify_arm:
        arm = args.verify_arm.strip().upper()
        locked = read_lock(lock_path)
        if arm not in locked.get("arms", {}):
            print(f"[ERROR] lock 中无该臂: {arm}", file=sys.stderr)
            return 2
        if arm not in results:
            print(f"[ERROR] 本次运行未执行该臂: {arm}", file=sys.stderr)
            return 2
        expected = locked["arms"][arm].get("artifact_sha256")
        actual = results[arm].artifact_sha256
        if expected != actual:
            print(f"[FAIL] 臂 {arm} 产物指纹不一致:\n  locked={expected}\n  current={actual}")
            return 1
        print(f"[OK] 臂 {arm} 产物指纹复现一致")
        return 0

    if args.verify:
        diffs = verify_lock(lock_path, lock)
        if diffs:
            print("[FAIL] 基线指纹不一致：")
            for d in diffs:
                print(f"  - {d}")
            return 1
        print("[OK] 基线指纹一致")
        return 0

    try:
        write_lock(
            lock, lock_path, force=args.force, reason=args.reason,
            allow_dirty=args.allow_dirty,
        )
    except (LockViolation, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 3

    summary = {
        "lock": str(lock_path).replace("\\", "/"),
        "arms": {
            n: {
                "n_in": r.n_in, "n_out": r.n_out,
                "artifact_sha256": r.artifact_sha256, "meta": r.meta,
            }
            for n, r in results.items()
        },
    }
    (report_dir / "arm_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # manifest 只溯源不判等：记录机器/依赖/模型，不入库、不参与 verify（ADR D7）
    manifest_path = write_manifest(
        build_manifest(
            root=Path.cwd(),
            profile=args.profile,
            arms=arms_wanted,
            models={
                "judge": DEFAULT_EVALUATOR["judge_model"],
                "generate": os.environ.get("DEEPSEEK_MODEL") or None,
                "ollama": os.environ.get("OLLAMA_MODEL") or None,
            },
        ),
        report_dir / "run_manifest.json",
    )

    print(f"[OK] lock -> {lock_path}")
    print(f"[OK] manifest -> {manifest_path}")
    for n, r in results.items():
        print(f"  {n}: in={r.n_in} out={r.n_out} sha={r.artifact_sha256[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
