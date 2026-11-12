"""T1 基线冻结 CLI

用法（在仓库根目录执行）：
    python -m agent.benchmark.run --input agent/fixtures/public/base_v1.json --arms A0,A1
    python -m agent.benchmark.run --input ... --verify   # 只校验，不重写 lock

产出：
    {lock_dir}/baseline_lock.json   四元组指纹（只读，入库）
    {out}/report/arm_results.json   各臂产物指纹与规模（不入库）
"""

import argparse
import json
import sys
from pathlib import Path

from .arms import run_a0, run_a1
from .fingerprint import (
    input_fingerprint,
    evaluator_fingerprint,
    adapter_fingerprint,
    build_lock,
)
from .lock import LockViolation, write_lock, read_lock, verify_lock


DEFAULT_EVALUATOR = dict(
    rules_version="1.0.0",
    judge_prompt_version="v1",
    judge_model="deepseek-chat",
    rubric_version="1.0",
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


def _a1_config(seed: int, num_variants: int):
    """A1 的配置快照（与 arms.run_a1 实际使用的一致），用于配置指纹。"""
    from .arms import DEFAULT_A1_AUGMENTERS
    return {
        "strategy": "single", "min_steps": 1, "max_steps": 1,
        "seed": seed, "num_variants": num_variants,
        "target_roles": ["assistant"], "only_loss_true": True,
        "augmenters": DEFAULT_A1_AUGMENTERS,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.run")
    ap.add_argument("--input", required=True, help="输入对话 JSON/JSONL")
    ap.add_argument("--arms", default="A0",
                    help="默认只跑 A0（可独立运行）；A1 需主工程环境，须显式指定")
    ap.add_argument("--out", default="agent/benchmark/output")
    ap.add_argument("--lock-dir", default="agent/benchmark/baseline",
                    help="基线目录（入库）；--out 为运行产物（不入库）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-variants", type=int, default=1)
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--verify", action="store_true", help="只校验已有 lock（要求臂集合完全一致）")
    ap.add_argument("--verify-arm", default=None,
                    help="只校验单个臂的产物是否复现（用于可独立运行的环境，如仅 A0）")
    ap.add_argument("--force", action="store_true", help="覆盖 lock（需 --reason）")
    ap.add_argument("--reason", default="")
    args = ap.parse_args(argv)

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"[ERROR] 输入不存在: {in_path}", file=sys.stderr)
        return 2

    dialogues = _load(in_path)
    arms_wanted = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    results = {}
    if "A0" in arms_wanted:
        results["A0"] = run_a0(dialogues)
    if "A1" in arms_wanted:
        work = Path(args.out) / "work" / "A1"
        results["A1"] = run_a1(
            dialogues, work, seed=args.seed, num_variants=args.num_variants
        )

    arm_fps = {}
    for name, r in results.items():
        cfg = {"method": r.arm} if name == "A0" else _a1_config(args.seed, args.num_variants)
        from .fingerprint import config_fingerprint
        arm_fps[name] = {
            "config_sha256": config_fingerprint(cfg),
            "prompt_version": None,
            # 产物指纹：同输入同配置必然复现，可在无 LLM 环境中自证基线未漂移
            "artifact_sha256": r.artifact_sha256,
        }

    lock = build_lock(
        input_fp=input_fingerprint(in_path, root=Path.cwd()),
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
        write_lock(lock, lock_path, force=args.force, reason=args.reason)
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

    print(f"[OK] lock -> {lock_path}")
    for n, r in results.items():
        print(f"  {n}: in={r.n_in} out={r.n_out} sha={r.artifact_sha256[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
