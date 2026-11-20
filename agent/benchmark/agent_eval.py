"""Agent 臂评测（A2 / A3 / A3L / A3R / A4 / A4R）

在同一缺陷集上跑**完整决策循环**，产出每类缺陷的检出率、阻断率、干净样本干预率与成本。

六个臂，按单变量原则逐层加因子：

| 臂 | 决策策略 | 工具 | 回答的问题 |
|---|---|---|---|
| `A2`   | 无（单次 LLM 重写） | 无 | 隔离"工具循环"本身的净收益 |
| `A3`   | 规则（确定性） | 确定性校验（无判官） | 不用 LLM 能做到什么 |
| `A3L`  | LLM function calling | 确定性校验（无判官） | 仅换决策者，值不值 |
| `A3R`  | 规则（多轮修复） | 确定性校验 + rewrite_segment | 给规则策略补上修复工具后能走多远 |
| `A4`   | LLM function calling | 确定性校验 + 语义判官 | 语义能力带来的增量 |
| `A4R`  | LLM function calling | 确定性校验 + 语义判官 + rewrite_segment | LLM 决策拿到完整工具集后的上限 |

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

from ..eval.coherence import check_coherence
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
from .report import write_review_markdown

A2 = "A2"
A3 = "A3"
A3L = "A3L"
A3R = "A3R"
A4 = "A4"
A4R = "A4R"
ARMS = (A2, A3, A3L, A3R, A4, A4R)

ARM_DESCRIPTIONS = {
    A2: "单次 LLM 重写，无工具/校验/回退",
    A3: "规则策略 + 确定性工具",
    A3L: "LLM 决策 + 确定性工具",
    A3R: "规则多轮策略 + 确定性工具 + rewrite_segment",
    A4: "LLM 决策 + 确定性工具 + 语义判官",
    A4R: "LLM 决策 + 确定性工具 + 语义判官 + rewrite_segment",
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
    # 干净样本未跑完（异常）：同样属于"被干预"，不得美化成未误报
    clean_failed: int = 0
    # 检出后修复并复检通过的样本数（修复能力，区别于回退）
    repaired: int = 0
    failed: int = 0
    # 因预算/墙钟触顶而中止的样本数：宿主机待机等环境噪声会让它变高，
    # 单独计数才能避免把"环境中断"误读成"模型没检出"
    stopped: int = 0
    tokens: int = 0
    calls: int = 0
    cost_cny: float = 0.0
    elapsed_s: float = 0.0
    # 已修复样本的原始对照，供 --review 导出人工抽检；不进入 to_dict
    samples: List[Dict[str, Any]] = field(default_factory=list)

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

    def clean_intervention_rate(self) -> float:
        """干净样本干预率：被回退或被异常中断的干净样本占比。

        未跑完的样本按"被干预"计，避免失败样本被美化成"没误伤"（R7）。
        """
        if not self.clean_total:
            return 0.0
        return round((self.clean_flagged + self.clean_failed) / self.clean_total, 4)

    def repair_rate(self) -> float:
        """修复成功率 = 修复通过数 / 被标记数（设计文档 L3，目标 ≥ 0.70）。"""
        flagged = sum(r.get("detected", 0) for r in self.per_type.values())
        return round(self.repaired / flagged, 4) if flagged else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "description": ARM_DESCRIPTIONS.get(self.arm, ""),
            "per_type": OrderedDict(sorted(self.per_type.items())),
            "macro_recall_blocked": self.macro_recall("blocked"),
            "macro_recall_detected": self.macro_recall("detected"),
            "repaired": self.repaired,
            "repair_rate": self.repair_rate(),
            "clean_total": self.clean_total,
            "clean_flagged": self.clean_flagged,
            "clean_failed": self.clean_failed,
            "clean_intervention_rate": self.clean_intervention_rate(),
            "failed": self.failed,
            "stopped": self.stopped,
            "tokens": self.tokens,
            "calls": self.calls,
            "cost_cny": self.cost_cny,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def build_runtime(arm: str, *, client: Any = None, judge: Any = None,
                  run_id: str = "ablation", trace: Any = None):
    """按臂装配运行时；会先清空工具注册表，避免跨臂污染。"""
    if arm not in ARMS:
        raise ValueError(f"未知臂: {arm}（可选 {ARMS}）")

    ToolRegistry.clear()

    if arm == A3:
        register_all()
        return _runtime(RulePolicy(), trace=trace)

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
            trace=trace,
        )

    if arm == A3L:
        register_all()
        return _runtime(LLMPolicy(client, run_id=run_id, trace=trace), trace=trace)

    # A4 / A4R 都带语义判官
    register_all(judge=judge or LLMJudge(client=client, run_id=run_id))
    if arm == A4R:
        ToolRegistry.register(make_rewrite_segment(client, run_id=run_id))
        # 两次改写 + 每轮复检，工具调用数会明显高于 A4，预算同步放宽
        return _runtime(
            LLMPolicy(client, run_id=run_id, allow_rewrite=True, trace=trace),
            budget=Budget(
                max_iterations=20,
                max_tool_calls=24,
                max_tokens=20000,
                max_wall_clock_s=300.0,
            ),
            trace=trace,
        )

    return _runtime(LLMPolicy(client, run_id=run_id, trace=trace), trace=trace)


def _runtime(policy, budget: Optional[Budget] = None, trace: Any = None):
    from ..runtime.loop import AgentRuntime

    return AgentRuntime(policy, budget=budget, trace=trace)


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
    """解析 A2 模型输出。

    只采纳 assistant 的 content；system / user 与内部标注字段（如 loss）
    一律以原文为准。模型回显时丢字段或改动不可改的消息都不应被算作"改坏"。
    结构与原文对不上时判为 parse_error。
    """
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
    orig_msgs = original.get("messages") or []
    if not isinstance(msgs, list) or len(msgs) != len(orig_msgs) or not orig_msgs:
        return {"ok": False}

    merged: List[Dict[str, Any]] = []
    for orig, new in zip(orig_msgs, msgs):
        if not isinstance(new, dict) or new.get("role") != orig.get("role"):
            return {"ok": False}
        if orig.get("role") == "assistant":
            entry = dict(orig)
            content = new.get("content")
            if not isinstance(content, str):
                return {"ok": False}
            entry["content"] = content
            merged.append(entry)
        else:
            # 模型不得改 system / user；按 prompt 约定强制回原文
            merged.append(dict(orig))

    # 保留原始 dialogue 里的 extra 字段（scenario tag 等）
    out = dict(original)
    out["messages"] = merged
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
    trace: Any = None,
) -> ArmResult:
    """跑一个臂，返回聚合指标。

    Args:
        base: 干净对话列表（回退参照与误报测量都用它）
        defects: 缺陷记录，形如 {"dialogue", "defect": {"type": ...}, "base_index"}
        ledger: 可选 TokenLedger，用于取真实成本
        trace: 可选 TraceWriter，落盘决策链（A2 无循环，不使用）
    """
    # 账本按臂归集：每次运行前把当前臂名注入客户端（含路由客户端的两个子客户端）
    for c in (client, judge):
        setter = getattr(c, "set_arm", None)
        if callable(setter):
            setter(arm)

    if arm == A2:
        return _run_a2(base, defects, client=client, run_id=run_id, ledger=ledger)

    started = time.monotonic()
    runtime = build_runtime(arm, client=client, judge=judge, run_id=run_id, trace=trace)
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
        elif r.status == "accepted" and r.seen_issues:
            # 检出过问题、改写后复检通过：这才是"修复"，不是"回退"
            row["repaired"] = row.get("repaired", 0) + 1
            result.repaired += 1
            # 自动"修复通过"不等于语义正确，留原始对照给人工抽检
            result.samples.append({
                "key": dialogue_key(record["dialogue"]),
                "type": dtype,
                "status": r.status,
                "reason": r.reason,
                "seen_issues": list(r.seen_issues),
                "issues": list(r.issues),
                "reference": reference,
                "input": record["dialogue"],
                "final": r.dialogue,
            })

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
        elif r.status == "failed":
            result.clean_failed += 1

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
            result.clean_failed += 1
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
# L3 跨臂对比（建在 run_arm 之上，复用同一个 ledger 和数据）
# --------------------------------------------------------------------------

def compare_arms(
    arm_a: str,
    arm_b: str,
    base: List[dict],
    defects: List[dict],
    *,
    client: Any = None,
    run_id: str = "compare",
    ledger: Any = None,
    metric: str = "blocked",
    trace: Any = None,
) -> Dict[str, Any]:
    """对比两个臂在同一数据上的 L3 统计。

    Args:
        metric: "blocked"（默认，最严格门禁指标）| "detected"（检出指标）| "clean"（干净样本未被标记）

    Returns:
        包含两个 ArmResult、win_rate 和 mcnemar 结果的字典
    """
    # 收集 per-sample outcome：先跑 arm_a，再跑 arm_b，复用同一个 ledger 记账
    outcomes_a = _collect_outcomes(
        arm_a, base, defects, client=client, run_id=f"{run_id}_{arm_a}",
        ledger=ledger, metric=metric, trace=trace,
    )
    outcomes_b = _collect_outcomes(
        arm_b, base, defects, client=client, run_id=f"{run_id}_{arm_b}",
        ledger=ledger, metric=metric, trace=trace,
    )

    # 两边长度一致（同一数据），outcomes 顺序 = defects + base
    assert len(outcomes_a) == len(outcomes_b)

    from .l3_stats import win_rate, mcnemar
    wr = win_rate(outcomes_a, outcomes_b)
    mn = mcnemar(outcomes_a, outcomes_b)

    return {
        "arms": [arm_a, arm_b],
        "metric": metric,
        "n_total": len(outcomes_a),
        "win_rate": {
            "wins": wr.wins_a, "losses": wr.wins_b, "ties": wr.ties,
            "comparable": wr.n_comparable,
            "point": wr.point, "ci_lower": wr.ci_lower, "ci_upper": wr.ci_upper,
        },
        "mcnemar": {
            "chi2": mn.chi2, "p_value": mn.p_value,
            "method": mn.method,
            "n10": mn.n10, "n01": mn.n01, "n11": mn.n11, "n00": mn.n00,
        },
        "significance": "显著" if mn.p_value < 0.05 else "不显著",
        "wilson_interpretation": (
            f"{arm_a} 对 {arm_b} 的 {metric} 胜率 Wilson 95% CI "
            f"[{wr.ci_lower:.4f}, {wr.ci_upper:.4f}]，"
            f"点估计 {wr.point:.4f}（排除 {wr.ties} 平局）"
        ),
    }


def _collect_outcomes(
    arm: str,
    base: List[dict],
    defects: List[dict],
    *,
    client: Any,
    run_id: str,
    ledger: Any,
    metric: str,
    trace: Any = None,
) -> List[bool]:
    """对每个 defect 样本 + 每个 clean 样本产出一个二进制 outcome。

    metric='blocked' → defect 样本返回 arm 是否 fallback；clean 样本返回 arm 是否没误报（True=好）
    metric='detected' → defect 样本返回 arm 是否检出；clean 样本同 blocked
    metric='no_leak' → defect 样本返回缺陷是否未泄露（回退或修复通过）；clean 样本同 blocked
    metric='clean'   → 只跑 clean 样本，返回是否没被标记

    """
    # 不走 run_arm（它是 aggregate only），直接调底层路径收集 per-sample
    outcomes: List[bool] = []

    if arm == A2:
        return _collect_outcomes_a2(base, defects, client=client, run_id=run_id,
                                     ledger=ledger, metric=metric)

    runtime = build_runtime(arm, client=client, run_id=run_id, trace=trace)
    # no_leak 用独立评估器复检最终输出，不采信 agent 自身的校验结论
    leak_judge = None
    if metric == "no_leak" and client is not None:
        from ..eval.judge import LLMJudge

        leak_judge = LLMJudge(client=client, run_id=f"{run_id}_leak")

    # defect samples
    for record in defects:
        reference = base[record["base_index"]] if base else None
        r = runtime.run(record["dialogue"], reference=reference)
        if metric == "detected":
            outcomes.append(bool(r.issues or r.seen_issues))
        elif metric == "no_leak":
            # 缺陷未泄露：独立复检最终输出
            # 结构/实体看规则对比，重复空转/空回复看连贯性规则，D1 看判官
            violations = (check_consistency(before=reference, after=r.dialogue)
                          if reference else [])
            violations += check_coherence(r.dialogue, judge=leak_judge)
            outcomes.append(not violations)
        else:  # blocked
            outcomes.append(r.status == "fallback")

    # clean samples
    for dialogue in base:
        r = runtime.run(dialogue)
        # clean 样本：metric='blocked'/'detected' 都看"没被干预"
        # 被回退、异常中断或报出 issues 都算干预（失败不得美化成未误报）
        outcomes.append(
            r.status not in ("fallback", "failed") and not r.issues
        )

    return outcomes


def _collect_outcomes_a2(
    base: List[dict],
    defects: List[dict],
    *,
    client: Any,
    run_id: str,
    ledger: Any,
    metric: str,
) -> List[bool]:
    """A2 专用：不经过 AgentRuntime，直接调 run_a2_dialogue。"""
    outcomes: List[bool] = []

    for record in defects:
        reference = base[record["base_index"]] if base else None
        # A2 没有回退：只有改写后与干净参照一致，才算缺陷未泄露
        r = run_a2_dialogue(client, record["dialogue"], run_id=run_id + "_d")
        fidelity = (check_consistency(before=reference, after=r["dialogue"])
                    if reference else [])
        outcomes.append(not r["parse_error"] and not fidelity)

    for dialogue in base:
        r = run_a2_dialogue(client, dialogue, run_id=run_id + "_c")
        fidelity = check_consistency(before=dialogue, after=r["dialogue"])
        outcomes.append(not bool(fidelity) and not r["parse_error"])

    return outcomes


def _load_public(root: Path):
    """加载公开轨 fixture：干净对话列表 + 缺陷记录列表。"""
    pub = root / "agent" / "fixtures" / "public"
    base = json.loads((pub / "base_v1.json").read_text(encoding="utf-8"))
    lines = (pub / "defects_v1.jsonl").read_text(encoding="utf-8").strip().split("\n")
    defects = [json.loads(line) for line in lines if line.strip()]
    return base, defects


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.agent_eval")
    ap.add_argument("--arm", choices=list(ARMS), help="单臂评测")
    ap.add_argument("--compare", default="",
                    help="L3 跨臂对比：\"A3,A2\" 或 \"A4,A3L\" 格式")
    ap.add_argument("--live", action="store_true", help="在线臂：需要 .env 中的密钥")
    ap.add_argument("--limit", type=int, default=0, help="每类最多取 N 条（调试用）")
    ap.add_argument("--metric", default="blocked", choices=["blocked", "detected", "clean", "no_leak"],
                    help="L3 对比用的 outcome 定义，默认 blocked（最严格门禁）")
    ap.add_argument("--out", default="", help="结果 JSON 路径；默认打印到标准输出")
    ap.add_argument("--trace", default="",
                    help="决策链 JSONL 输出路径；用于定位 policy_error / 循环等失败原因")
    ap.add_argument("--review", default="",
                    help="人工抽检清单（Markdown）输出路径；只导出 repaired 样本")
    args = ap.parse_args(argv)

    if not args.arm and not args.compare:
        print("[ERROR] 必须指定 --arm 或 --compare", file=sys.stderr)
        return 2
    if args.arm and args.compare:
        print("[ERROR] --arm 和 --compare 不能同时用", file=sys.stderr)
        return 2

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

    # 确定需要 live 的臂
    need_live: set[str] = set()
    if args.arm:
        need_live.add(args.arm)
    if args.compare:
        a, b = args.compare.split(",")
        need_live.update([a.strip(), b.strip()])
    has_online = any(arm != A3 for arm in need_live)

    client = ledger = None
    model_name = judge_model_name = None
    if has_online:
        if not args.live:
            print(f"[ERROR] 包含在线臂，需要 --live（会消耗 token）", file=sys.stderr)
            return 2
        from ..runtime.config import LLMSettings, load_dotenv
        from ..runtime.deepseek import build_stage_clients
        from ..runtime.ledger import TokenLedger

        load_dotenv(root / ".env")
        settings = LLMSettings.from_env()
        if not settings.has_key():
            print("[ERROR] 未找到 DEEPSEEK_API_KEY", file=sys.stderr)
            return 2
        ledger = TokenLedger(root / "logs" / "agent" / f"ledger_compare.jsonl")
        client = build_stage_clients(ledger=ledger)
        model_name = settings.model
        judge_model_name = judge_settings().model

    # 每次运行的账本标识必须唯一，否则多次尝试会混进同一 run_id
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    trace = None
    if args.trace:
        from ..runtime.trace import TraceWriter

        trace = TraceWriter(args.trace)

    if args.compare:
        arm_a, arm_b = [s.strip() for s in args.compare.split(",")]
        for arm in (arm_a, arm_b):
            if arm not in ARMS:
                print(f"[ERROR] 未知臂: {arm}", file=sys.stderr)
                return 2
        payload = compare_arms(arm_a, arm_b, base, defects, client=client,
                               run_id=f"compare_{arm_a}_vs_{arm_b}_{stamp}",
                               ledger=ledger, metric=args.metric, trace=trace)
        if ledger:
            ledger.close()
        if trace:
            trace.close()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(payload.get("wilson_interpretation", ""))
        print(f"  p={payload['mcnemar']['p_value']} ({payload['mcnemar']['method']}), "
              f"{payload['significance']}")
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
            print(f"[OK] {arm_a} vs {arm_b} -> {out}")
        else:
            print(text)
        return 0

    # 单臂评测路径
    result = run_arm(args.arm, base, defects, client=client, ledger=ledger,
                     run_id=f"ablation_{args.arm}_{stamp}", trace=trace)
    if ledger:
        ledger.close()
    if trace:
        trace.close()

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
            judge_model=judge_model_name if args.arm in (A4, A4R) else "none",
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
                        "judge": judge_model_name if args.arm in (A4, A4R) else None},
                extra={"limit": args.limit, "n_defects": len(defects), "n_base": len(base)},
            ),
            out.parent / f"manifest_{args.arm}.json",
        )
        print(f"[OK] {args.arm} -> {out}")
    else:
        print(text)

    if args.review:
        review = write_review_markdown(args.review, result.samples,
                                       arm=args.arm, generated_at=stamp)
        print(f"[OK] 抽检清单 -> {review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
