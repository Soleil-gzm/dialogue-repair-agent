"""Agent 臂评测（A2 / A3 / A3L / A3R / A4）

在同一缺陷集上跑**完整决策循环**，产出每类缺陷的检出率、阻断率、干净样本误报与成本。

五个臂，按单变量原则逐层加因子：

| 臂 | 决策策略 | 工具 | 回答的问题 |
|---|---|---|---|
| `A2`   | 无（单次 LLM 重写） | 无 | 隔离"工具循环"本身的净收益 |
| `A3`   | 规则（确定性） | 确定性校验（无判官） | 不用 LLM 能做到什么 |
| `A3L`  | LLM function calling | 确定性校验（无判官） | 仅换决策者，值不值 |
| `A3R`  | 规则（多轮修复） | 确定性校验 + rewrite_segment | 给规则策略补上修复工具后能走多远 |
| `A4`   | LLM function calling | 确定性校验 + 语义判官 | 语义能力带来的增量 |

口径：
- **detected** = 循环期间曾经记录到违规（说明看见问题了，修复成功也算）
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

from ..eval.judge import JUDGE_VERSION, RUBRIC_VERSION, LLMJudge
from ..eval.rules import RULES_VERSION, check_consistency
from ..runtime.budget import Budget
from ..runtime.config import judge_settings
from ..runtime.policy import LLMPolicy, RulePolicy
from ..runtime.registry import ToolRegistry
from ..runtime.state import dialogue_key
from ..tools import register_all
from ..tools.rewrite_segment import REWRITE_SEGMENT, make_rewrite_segment
from .fingerprint import evaluator_fingerprint, input_fingerprint
from .manifest import build_manifest, write_manifest

A2 = "A2"
A3 = "A3"
A3L = "A3L"
A3R = "A3R"
A4 = "A4"
ARMS = (A2, A3, A3L, A3R, A4)

ARM_DESCRIPTIONS = {
    A2: "单次 LLM 重写，无工具/校验/回退",
    A3: "规则策略 + 确定性工具",
    A3L: "LLM 决策 + 确定性工具",
    A3R: "规则多轮策略 + 确定性工具 + rewrite_segment",
    A4: "LLM 决策 + 确定性工具 + 语义判官",
}

# A2 单次重写 prompt：给模型一段对话，让它修正 assistant 回复的答非所问问题
# 不提供工具，直接输出修正后的对话 JSON
A2_REWRITE_PROMPT = """你是一个对话修复器。下面的对话中，assistant 的某些回复可能答非所问、逻辑不通或内容有问题。
请只修改 assistant 的回复，使它们与用户的问题真正对应。
system 消息绝对不能修改。user 消息绝对不能修改。只改 assistant 的 content 字段。

输出严格为 JSON，格式：
{"messages": [
  {"role": "system", "content": "... 原样返回 ..."},
  {"role": "user", "content": "... 原样返回 ..."},
  {"role": "assistant", "content": "... 可修改 ..."},
  ...
]}

如果对话没问题，原样返回即可。"""


@dataclass
class ArmResult:
    arm: str
    per_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    clean_total: int = 0
    clean_flagged: int = 0
    failed: int = 0
    # 因预算/墙钟触顶而中止的样本数：宿主机待机等环境噪声会让它变高，
    # 单独计数才能避免把"环境中断"误读成"模型没检出"
    stopped: int = 0
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
            "stopped": self.stopped,
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

    if arm == A3R:
        register_all()
        ToolRegistry.register(make_rewrite_segment(client, run_id=run_id))
        return _runtime(
            RulePolicy(rewrite_tool=REWRITE_SEGMENT),
            budget=Budget(
                max_iterations=20,
                max_tool_calls=24,
                max_tokens=20000,
                max_wall_clock_s=300.0,
            ),
        )

    if arm == A3L:
        register_all()
    else:
        register_all(judge=judge or LLMJudge(client=client, run_id=run_id))

    return _runtime(LLMPolicy(client, run_id=run_id))


def _runtime(policy, budget: Optional[Budget] = None):
    from ..runtime.loop import AgentRuntime

    return AgentRuntime(policy, budget=budget)


# --------------------------------------------------------------------------
# A2：单次 LLM 重写，不经过 AgentRuntime 循环
# --------------------------------------------------------------------------

def run_a2_dialogue(client: Any, dialogue: Dict[str, Any],
                    run_id: str = "a2",
                    reference: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A2 单次 LLM 重写——一次 chat，无条件接受结果。

    Returns:
        {"ok": bool, "dialogue": dict, "tokens": int, "parse_error": bool}
    """
    messages = [
        {"role": "system", "content": A2_REWRITE_PROMPT},
        {"role": "user", "content": json.dumps(dialogue, ensure_ascii=False)},
    ]
    resp = client.chat(
        messages,
        temperature=0.3,
        max_tokens=2048,
        stage="generate",
        run_id=run_id,
        dialogue_key=dialogue_key(dialogue),
    )
    out_tokens = resp.total_tokens
    parsed = _parse_a2_output(resp.text, dialogue)
    return {
        "ok": parsed["ok"],
        "dialogue": parsed.get("dialogue") or dialogue,
        "tokens": out_tokens,
        "parse_error": not parsed["ok"],
    }


def _parse_a2_output(text: str, original: Dict[str, Any]) -> Dict[str, Any]:
    """解析 A2 模型输出——宽松：找到第一个 {...} 块就尝试 parse。"""
    text = (text or "").strip()
    # 直接 parse
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        # 兜底：从 ```json ... ``` 或 ``` ... ``` 中提取
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except (TypeError, ValueError):
                return {"ok": False}
        else:
            return {"ok": False}
    msgs = data.get("messages") or []
    if not isinstance(msgs, list) or len(msgs) < 2:
        return {"ok": False}
    # 保留原始 dialogue 里的 extra 字段（scenario tag 等）
    out = dict(original)
    out["messages"] = msgs
    return {"ok": True, "dialogue": out}


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
    if arm == A2:
        return _run_a2(base, defects, client=client, run_id=run_id, ledger=ledger)

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
        if r.status == "stopped":
            result.stopped += 1
        if r.issues or r.seen_issues:
            row["detected"] += 1
        if r.status == "fallback":
            row["blocked"] += 1

    for dialogue in base:
        result.clean_total += 1
        r = runtime.run(dialogue)
        result.tokens += r.tokens
        if r.status == "failed":
            result.failed += 1
        if r.status == "stopped":
            result.stopped += 1
        if r.status == "fallback":
            result.clean_flagged += 1

    if ledger:
        result.tokens = ledger.total_tokens() - tokens_before
        result.cost_cny = round(ledger.total_cost() - cost_before, 8)
        result.calls = len(ledger.entries) - calls_before

    result.elapsed_s = time.monotonic() - started
    return result


def _run_a2(
    base: List[dict],
    defects: List[dict],
    *,
    client: Any,
    run_id: str,
    ledger: Any,
) -> ArmResult:
    """A2 专用路径：不经过 AgentRuntime 决策循环，一次 LLM 重写 + 无条件接受。"""
    started = time.monotonic()
    result = ArmResult(arm=A2)
    tokens_before = ledger.total_tokens() if ledger else 0
    cost_before = ledger.total_cost() if ledger else 0.0
    calls_before = len(ledger.entries) if ledger else 0

    for record in defects:
        dtype = (record.get("defect") or {}).get("type") or "unknown"
        row = result.per_type.setdefault(dtype, {"total": 0, "detected": 0, "blocked": 0})
        row["total"] += 1

        reference = base[record["base_index"]] if base else None
        # pre-A2 违规：用 reference（干净原文）和 dialogue（缺陷版）比一致性
        pre_violations = check_consistency(
            before=reference or {}, after=record["dialogue"],
        ) if reference else []

        r = run_a2_dialogue(client, record["dialogue"], run_id=f"{run_id}_a2")
        result.tokens += r["tokens"]

        if r["parse_error"]:
            result.failed += 1
        if pre_violations:
            row["detected"] += 1
        # A2 没有 fallback：blocked 永远为 0

    for dialogue in base:
        result.clean_total += 1
        r = run_a2_dialogue(client, dialogue, run_id=f"{run_id}_a2c")
        result.tokens += r["tokens"]

        if r["parse_error"]:
            result.failed += 1
            continue
        # 干净样本被 A2 改坏：post-A2 与原文的一致性校验失败
        fidelity = check_consistency(before=dialogue, after=r["dialogue"])
        if fidelity:
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
    pub = root / "agent" / "fixtures" / "public"
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
    model_name = None
    if args.arm != A3:
        if not args.live:
            print(f"[ERROR] 臂 {args.arm} 需要 --live（会消耗 token）", file=sys.stderr)
            return 2
        from ..runtime.config import LLMSettings, load_dotenv
        from ..runtime.deepseek import build_stage_clients
        from ..runtime.ledger import TokenLedger

        load_dotenv(root / ".env")
        settings = LLMSettings.from_env()
        if not settings.has_key():
            print("[ERROR] 未找到 DEEPSEEK_API_KEY", file=sys.stderr)
            return 2
        ledger = TokenLedger(root / "logs" / "agent" / f"ledger_{args.arm}.jsonl")
        # 所有在线臂统一走 StageRoutedClient：
        # - A3R rewrite 走 generate 侧（stage != "judge"）
        # - A4 的 LLMJudge 以 stage="judge" 调用 → 路由到 JUDGE_MODEL 配置
        # - A3L 只有决策调用 → 走 generate 侧
        client = build_stage_clients(ledger=ledger)
        model_name = settings.model
        judge_model_name = judge_settings().model

    result = run_arm(args.arm, base, defects, client=client, ledger=ledger,
                     run_id=f"ablation_{args.arm}")
    if ledger:
        ledger.close()

    payload = result.to_dict()
    # 结果必须自带出处：输入与评估器指纹进结果，环境信息单独写 manifest（不判等）
    payload["fingerprint"] = {
        "input": {
            "base": input_fingerprint(pub / "base_v1.json", root=root),
            "defects": input_fingerprint(pub / "defects_v1.jsonl", root=root),
        },
        "evaluator": evaluator_fingerprint(
            rules_version=RULES_VERSION,
            judge_prompt_version=JUDGE_VERSION,
            judge_model=judge_model_name if args.arm == A4 else "none",
            rubric_version=RUBRIC_VERSION,
            judge_temperature=0.0,
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        write_manifest(
            build_manifest(
                root=root,
                profile="ablation",
                arms=[args.arm],
                models={"generate": model_name,
                        "judge": judge_model_name if args.arm == A4 else None},
                extra={"limit": args.limit, "n_defects": len(defects), "n_base": len(base)},
            ),
            out.parent / f"manifest_{args.arm}.json",
        )
        print(f"[OK] {args.arm} -> {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
