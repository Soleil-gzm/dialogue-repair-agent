"""Agent 臂评测（A3 / A3L / A4）

在同一缺陷集上跑**完整决策循环**，产出每类缺陷的检出率、阻断率、干净样本误报与成本。

三个臂，按单变量原则逐层加因子：

| 臂 | 决策策略 | 工具 | 回答的问题 |
|---|---|---|---|
| `A3`   | 规则（确定性） | 确定性校验（无判官） | 不用 LLM 能做到什么 |
| `A3L`  | LLM function calling | 确定性校验（无判官） | 仅换决策者，值不值 |
| `A4`   | LLM function calling | 确定性校验 + 语义判官 | 语义能力带来的增量 |

口径：
- **detected** = 跑完循环后存在违规记录（说明看见问题了）
- **blocked**  = 最终回退原文（说明真的挡住并保住了数据）
- 干净样本的 blocked 即误报

评测器不依赖具体模型：在线臂由调用方注入 LLMClient，离线臂（A3）无需密钥。
"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..eval.judge import LLMJudge
from ..runtime.policy import LLMPolicy, RulePolicy
from ..runtime.registry import ToolRegistry
from ..tools import register_all

A3 = "A3"
A3L = "A3L"
A4 = "A4"
ARMS = (A3, A3L, A4)

ARM_DESCRIPTIONS = {
    A3: "规则策略 + 确定性工具",
    A3L: "LLM 决策 + 确定性工具",
    A4: "LLM 决策 + 确定性工具 + 语义判官",
}


@dataclass
class ArmResult:
    arm: str
    per_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    clean_total: int = 0
    clean_flagged: int = 0
    failed: int = 0
    tokens: int = 0
    calls: int = 0
    cost_cny: float = 0.0
    elapsed_s: float = 0.0

    # ------------------------------------------------------------------ 指标
    def recall(self, defect_type: str, key: str = "blocked") -> float:
        row = self.per_type.get(defect_type) or {}
        total = row.get("total", 0)
        return round(row.get(key, 0) / total, 4) if total else 0.0

    def macro_recall(self, key: str = "blocked") -> float:
        rows = [r for r in self.per_type.values() if r.get("total")]
        if not rows:
            return 0.0
        return round(sum(r.get(key, 0) / r["total"] for r in rows) / len(rows), 4)

    def clean_fp_rate(self) -> float:
        return round(self.clean_flagged / self.clean_total, 4) if self.clean_total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "description": ARM_DESCRIPTIONS.get(self.arm, ""),
            "per_type": OrderedDict(sorted(self.per_type.items())),
            "macro_recall_blocked": self.macro_recall("blocked"),
            "macro_recall_detected": self.macro_recall("detected"),
            "clean_total": self.clean_total,
            "clean_flagged": self.clean_flagged,
            "clean_fp_rate": self.clean_fp_rate(),
            "failed": self.failed,
            "tokens": self.tokens,
            "calls": self.calls,
            "cost_cny": self.cost_cny,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def build_runtime(arm: str, *, client: Any = None, judge: Any = None,
                  run_id: str = "ablation"):
    """按臂装配运行时；会先清空工具注册表，避免跨臂污染。"""
    if arm not in ARMS:
        raise ValueError(f"未知臂: {arm}（可选 {ARMS}）")

    ToolRegistry.clear()

    if arm == A3:
        register_all()
        return _runtime(RulePolicy())

    if client is None:
        raise ValueError(f"臂 {arm} 需要注入 LLMClient")

    if arm == A3L:
        register_all()
    else:
        register_all(judge=judge or LLMJudge(client=client, run_id=run_id))

    return _runtime(LLMPolicy(client, run_id=run_id))


def _runtime(policy):
    from ..runtime.loop import AgentRuntime

    return AgentRuntime(policy)


def run_arm(
    arm: str,
    base: List[dict],
    defects: List[dict],
    *,
    client: Any = None,
    judge: Any = None,
    run_id: str = "ablation",
    ledger: Any = None,
) -> ArmResult:
    """跑一个臂，返回聚合指标。

    Args:
        base: 干净对话列表（回退参照与误报测量都用它）
        defects: 缺陷记录，形如 {"dialogue", "defect": {"type": ...}, "base_index"}
        ledger: 可选 TokenLedger，用于取真实成本
    """
    started = time.monotonic()
    runtime = build_runtime(arm, client=client, judge=judge, run_id=run_id)
    result = ArmResult(arm=arm)
    tokens_before = ledger.total_tokens() if ledger else 0
    cost_before = ledger.total_cost() if ledger else 0.0
    calls_before = len(ledger.entries) if ledger else 0

    for record in defects:
        dtype = (record.get("defect") or {}).get("type") or "unknown"
        row = result.per_type.setdefault(dtype, {"total": 0, "detected": 0, "blocked": 0})
        row["total"] += 1

        reference = base[record["base_index"]] if base else None
        r = runtime.run(record["dialogue"], reference=reference)

        result.tokens += r.tokens
        if r.status == "failed":
            result.failed += 1
        if r.issues:
            row["detected"] += 1
        if r.status == "fallback":
            row["blocked"] += 1

    for dialogue in base:
        result.clean_total += 1
        r = runtime.run(dialogue)
        result.tokens += r.tokens
        if r.status == "failed":
            result.failed += 1
        if r.status == "fallback":
            result.clean_flagged += 1

    if ledger:
        result.tokens = ledger.total_tokens() - tokens_before
        result.cost_cny = round(ledger.total_cost() - cost_before, 8)
        result.calls = len(ledger.entries) - calls_before

    result.elapsed_s = time.monotonic() - started
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_public(root: Path):
    pub = root / "agent" / "fixtures" / "public"
    base = json.loads((pub / "base_v1.json").read_text(encoding="utf-8"))
    lines = (pub / "defects_v1.jsonl").read_text(encoding="utf-8").strip().split("\n")
    defects = [json.loads(l) for l in lines if l.strip()]
    return base, defects


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.agent_eval")
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--live", action="store_true", help="在线臂：需要 .env 中的密钥")
    ap.add_argument("--limit", type=int, default=0, help="每类最多取 N 条（调试用）")
    ap.add_argument("--out", default="", help="结果 JSON 路径；默认打印到标准输出")
    args = ap.parse_args(argv)

    root = Path.cwd()
    base, defects = _load_public(root)
    if args.limit > 0:
        kept, counts = [], {}
        for d in defects:
            t = d["defect"]["type"]
            counts[t] = counts.get(t, 0) + 1
            if counts[t] <= args.limit:
                kept.append(d)
        defects = kept

    client = judge = ledger = None
    if args.arm != A3:
        if not args.live:
            print(f"[ERROR] 臂 {args.arm} 需要 --live（会消耗 token）", file=sys.stderr)
            return 2
        from ..runtime.config import LLMSettings, load_dotenv
        from ..runtime.deepseek import DeepSeekClient
        from ..runtime.ledger import TokenLedger

        load_dotenv(root / ".env")
        settings = LLMSettings.from_env()
        if not settings.has_key():
            print("[ERROR] 未找到 DEEPSEEK_API_KEY", file=sys.stderr)
            return 2
        ledger = TokenLedger(root / "logs" / "agent" / f"ledger_{args.arm}.jsonl")
        client = DeepSeekClient(settings=settings, ledger=ledger)

    result = run_arm(args.arm, base, defects, client=client, ledger=ledger,
                     run_id=f"ablation_{args.arm}")
    if ledger:
        ledger.close()

    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"[OK] {args.arm} -> {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
