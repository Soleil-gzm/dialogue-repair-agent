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

from ..eval.coherence import C0_JUDGE_ERROR, check_coherence
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

# A2 单次重写的采样温度：固定值便于复现，并写入 run/manifest 供追溯
A2_TEMPERATURE = 0.3


@dataclass
class ArmResult:
    arm: str
    per_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    clean_total: int = 0
    clean_flagged: int = 0
    # 干净样本未跑完（异常）：同样属于"被干预"，不得美化成未误报
    clean_failed: int = 0
    # 干净样本因预算/墙钟中止：环境噪声，但也不能算"没误伤"
    clean_stopped: int = 0
    # 检出后修复并复检通过的样本数（修复能力，区别于回退）
    repaired: int = 0
    failed: int = 0
    # 因预算/墙钟触顶而中止的样本数：宿主机待机等环境噪声会让它变高，
    # 单独计数才能避免把"环境中断"误读成"模型没检出"
    stopped: int = 0
    # 判官故障样本数：基础设施错误不能算作模型检出/阻断，需单独隔离
    judge_error: int = 0
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

    def judge_error_rate(self) -> float:
        denominator = sum(r.get("total", 0) for r in self.per_type.values())
        denominator += self.clean_total
        return round(self.judge_error / denominator, 4) if denominator else 0.0

    def valid(self) -> bool:
        """判官故障超过 5% 时整轮结果不可采信。"""
        return self.judge_error_rate() <= 0.05

    def clean_intervention_rate(self) -> float:
        """干净样本干预率：被回退、被异常中断或被预算中止的干净样本占比。

        未跑完的样本按"被干预"计，避免失败样本被美化成"没误伤"（R7）。
        """
        if not self.clean_total:
            return 0.0
        intervened = self.clean_flagged + self.clean_failed + self.clean_stopped
        return round(intervened / self.clean_total, 4)

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
            "clean_stopped": self.clean_stopped,
            "clean_intervention_rate": self.clean_intervention_rate(),
            "failed": self.failed,
            "stopped": self.stopped,
            "judge_error": self.judge_error,
            "judge_error_rate": self.judge_error_rate(),
            "valid": self.valid(),
            "tokens": self.tokens,
            "calls": self.calls,
            "cost_cny": self.cost_cny,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _judge_failed(result: Any) -> bool:
    """样本是否因判官故障被污染。"""
    issues = list(getattr(result, "seen_issues", []) or [])
    issues += list(getattr(result, "issues", []) or [])
    return any((v or {}).get("rule") == C0_JUDGE_ERROR for v in issues)


def is_clean_intervention(result: Any) -> bool:
    """干净样本是否被干预。

    四种都属于干预，不能只算 fallback：
    - fallback / failed / stopped；
    - 报出过任何 issue（含误报后改写成功：issues 被清空但 seen_issues 仍在）；
    - 判官故障（无法证明未误伤）。
    """
    return (
        getattr(result, "status", "") in ("fallback", "failed", "stopped")
        or bool(getattr(result, "issues", None))
        or bool(getattr(result, "seen_issues", None))
        or _judge_failed(result)
    )


def _assistant_contents(dialogue: Dict[str, Any]) -> List[str]:
    return [str(m.get("content", "")) for m in (dialogue or {}).get("messages") or []
            if m.get("role") == "assistant"]


def a2_clean_intervened(parsed: Dict[str, Any], original: Dict[str, Any]) -> bool:
    """A2 干净样本是否被干预：解析失败，或 assistant 内容被改动。"""
    if not parsed.get("ok"):
        return True
    return _assistant_contents(parsed["dialogue"]) != _assistant_contents(original)


def build_runtime(arm: str, *, client: Any = None, judge: Any = None,
                  run_id: str = "ablation", trace: Any = None):
    """按臂装配运行时；会先清空工具注册表，避免跨臂污染。"""
    if arm not in ARMS:
        raise ValueError(f"未知臂: {arm}（可选 {ARMS}）")
    if arm == A2:
        raise ValueError("A2 走单次重写路径（run_a2_dialogue），不使用 AgentRuntime")

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
                    reference: Optional[Dict[str, Any]] = None,
                    temperature: float = A2_TEMPERATURE) -> Dict[str, Any]:
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
        temperature=temperature,
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
    tokens_before = ledger.total_tokens(include_cached=False) if ledger else 0
    cost_before = ledger.total_cost() if ledger else 0.0
    calls_before = ledger.n_calls(include_cached=False) if ledger else 0

    for record in defects:
        dtype = (record.get("defect") or {}).get("type") or "unknown"
        row = result.per_type.setdefault(dtype, {"total": 0, "detected": 0, "blocked": 0})
        row["total"] += 1

        reference = base[record["base_index"]] if base else None
        r = runtime.run(record["dialogue"], reference=reference)

        result.tokens += r.tokens
        judge_failed = _judge_failed(r)
        if judge_failed:
            result.judge_error += 1
        if r.status == "failed":
            result.failed += 1
        if r.status == "stopped":
            result.stopped += 1
        # 判官故障是基础设施问题，不得计成"模型检出/阻断"
        if judge_failed:
            continue
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
        if _judge_failed(r):
            result.judge_error += 1
        if r.status == "failed":
            result.failed += 1
            result.clean_failed += 1
        elif r.status == "stopped":
            result.stopped += 1
            result.clean_stopped += 1
        elif is_clean_intervention(r):
            result.clean_flagged += 1

    if ledger:
        result.tokens = ledger.total_tokens(include_cached=False) - tokens_before
        result.cost_cny = round(ledger.total_cost() - cost_before, 8)
        result.calls = ledger.n_calls(include_cached=False) - calls_before

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
    tokens_before = ledger.total_tokens(include_cached=False) if ledger else 0
    cost_before = ledger.total_cost() if ledger else 0.0
    calls_before = ledger.n_calls(include_cached=False) if ledger else 0

    for record in defects:
        dtype = (record.get("defect") or {}).get("type") or "unknown"
        row = result.per_type.setdefault(dtype, {"total": 0, "detected": 0, "blocked": 0})
        row["total"] += 1

        reference = base[record["base_index"]] if base else None
        # pre-A2 违规：噪声输入是否已能被确定性检查看见（不含 LLM 判官）
        pre_violations = check_consistency(
            before=reference or {}, after=record["dialogue"],
        ) if reference else []
        pre_violations += check_coherence(record["dialogue"])

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
        # 干净样本被 A2 动过（哪怕仍通过一致性）就算干预
        if a2_clean_intervened(r, dialogue):
            result.clean_flagged += 1

    if ledger:
        result.tokens = ledger.total_tokens(include_cached=False) - tokens_before
        result.cost_cny = round(ledger.total_cost() - cost_before, 8)
        result.calls = ledger.n_calls(include_cached=False) - calls_before

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
    judge: Any = None,
    run_id: str = "compare",
    ledger: Any = None,
    metric: str = "blocked",
    tie_rule: str = "exclude",
    trace: Any = None,
) -> Dict[str, Any]:
    """对比两个臂在同一数据上的 L3 统计。

    Args:
        metric: "blocked"（默认，最严格门禁指标）| "detected"（检出指标）| "clean"（干净样本未被标记）
        tie_rule: 平局处理 —— "exclude"（排除出胜率分母）| "count_half"（各计 0.5）

    Returns:
        包含两个 ArmResult、win_rate 和 mcnemar 结果的字典
    """
    if metric not in ("blocked", "detected", "clean", "no_leak"):
        raise ValueError(f"未知 metric: {metric}")

    # 收集 per-sample outcome：先跑 arm_a，再跑 arm_b，复用同一个 ledger 记账
    outcomes_a = _collect_outcomes(
        arm_a, base, defects, client=client, judge=judge, run_id=f"{run_id}_{arm_a}",
        ledger=ledger, metric=metric, trace=trace,
    )
    outcomes_b = _collect_outcomes(
        arm_b, base, defects, client=client, judge=judge, run_id=f"{run_id}_{arm_b}",
        ledger=ledger, metric=metric, trace=trace,
    )

    # 两边长度一致（同一数据），outcomes 顺序 = defects + base
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError(
            f"两个臂的 outcome 长度不一致: {arm_a}={len(outcomes_a)}, "
            f"{arm_b}={len(outcomes_b)}"
        )

    from .l3_stats import win_rate, mcnemar
    wr = win_rate(outcomes_a, outcomes_b, tie_rule=tie_rule)
    mn = mcnemar(outcomes_a, outcomes_b)

    if wr.point is None:
        wilson_note = (
            f"{arm_a} 对 {arm_b} 的 {metric} 全部为平局（{wr.ties} 条），"
            "胜率不可比（N/A），不能解读为 0%"
        )
    else:
        tie_note = (
            f"平局各计 0.5，分母 {wr.n_comparable}"
            if tie_rule == "count_half"
            else f"排除 {wr.ties} 平局"
        )
        wilson_note = (
            f"{arm_a} 对 {arm_b} 的 {metric} 胜率 Wilson 95% CI "
            f"[{wr.ci_lower:.4f}, {wr.ci_upper:.4f}]，"
            f"点估计 {wr.point:.4f}（{tie_note}）"
        )

    return {
        "arms": [arm_a, arm_b],
        "metric": metric,
        "tie_rule": tie_rule,
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
        "wilson_interpretation": wilson_note,
    }


def _set_arm(obj: Any, arm: str) -> None:
    setter = getattr(obj, "set_arm", None)
    if callable(setter):
        setter(arm)


def _collect_outcomes(
    arm: str,
    base: List[dict],
    defects: List[dict],
    *,
    client: Any,
    judge: Any = None,
    run_id: str,
    ledger: Any,
    metric: str,
    trace: Any = None,
) -> List[bool]:
    """对每个 defect 样本 + 每个 clean 样本产出一个二进制 outcome（True=好）。

    metric='blocked'  → defect：是否 fallback；clean：是否未被干预
    metric='detected' → defect：是否检出；clean：是否未被干预
    metric='no_leak'  → defect：独立复检后缺陷是否未泄露；clean：是否未被干预
    metric='clean'    → 只跑 clean 样本，返回是否未被干预

    判官故障（C0_JUDGE_ERROR）不能算作检出或阻断，相关样本一律按 False 处理。
    """
    if arm == A2:
        return _collect_outcomes_a2(
            base, defects, client=client, judge=judge, run_id=run_id,
            metric=metric,
        )

    _set_arm(client, arm)
    _set_arm(judge, arm)

    runtime = build_runtime(arm, client=client, judge=judge, run_id=run_id, trace=trace)

    # no_leak 用独立评估器复检最终输出，不采信 agent 自身的校验结论。
    # 无判官时无法检测 D1 语义泄漏，直接报错而不是静默退化。
    leak_judge = None
    if metric == "no_leak":
        leak_judge = judge or (LLMJudge(client=client, run_id=f"{run_id}_leak")
                               if client is not None else None)
        if leak_judge is None:
            raise ValueError("metric=no_leak 需要判官；请注入 judge 或在线 client")

    outcomes: List[bool] = []

    # clean 指标只跑干净样本，避免把缺陷 outcome 混进来
    if metric != "clean":
        for record in defects:
            reference = base[record["base_index"]] if base else None
            r = runtime.run(record["dialogue"], reference=reference)
            judge_failed = _judge_failed(r)
            if metric == "detected":
                outcomes.append(bool(r.issues or r.seen_issues) and not judge_failed)
            elif metric == "no_leak":
                violations = (check_consistency(before=reference, after=r.dialogue)
                              if reference else [])
                violations += check_coherence(r.dialogue, judge=leak_judge)
                outcomes.append(not violations and not judge_failed)
            else:  # blocked
                outcomes.append(r.status == "fallback" and not judge_failed)

    for dialogue in base:
        r = runtime.run(dialogue)
        outcomes.append(not is_clean_intervention(r))

    return outcomes


def _collect_outcomes_a2(
    base: List[dict],
    defects: List[dict],
    *,
    client: Any,
    judge: Any = None,
    run_id: str,
    metric: str,
) -> List[bool]:
    """A2 专用：不经过 AgentRuntime，直接调 run_a2_dialogue。

    A2 没有 fallback，因此 blocked 恒为 False；detected 用确定性预检
    （一致性 + 连贯性规则，无判官）判断噪声是否可见；no_leak 才真正调用
    A2 改写并做独立复检。
    """
    _set_arm(client, A2)
    _set_arm(judge, A2)
    outcomes: List[bool] = []

    if metric == "blocked":
        # 无回退能力：defect 侧不需要花一次生成就能给出确定答案
        outcomes = [False] * len(defects)
    elif metric == "detected":
        # 检出只看噪声输入是否已能被确定性检查看见，不调用 A2
        for record in defects:
            reference = base[record["base_index"]] if base else None
            violations = (check_consistency(before=reference, after=record["dialogue"])
                          if reference else [])
            violations += check_coherence(record["dialogue"])
            outcomes.append(bool(violations))
    elif metric == "no_leak":
        leak_judge = judge or (LLMJudge(client=client, run_id=f"{run_id}_leak")
                               if client is not None else None)
        if leak_judge is None:
            raise ValueError("metric=no_leak 需要判官；请注入 judge 或在线 client")
        for record in defects:
            reference = base[record["base_index"]] if base else None
            r = run_a2_dialogue(client, record["dialogue"], run_id=run_id + "_d")
            violations = (check_consistency(before=reference, after=r["dialogue"])
                          if reference else [])
            violations += check_coherence(r["dialogue"], judge=leak_judge)
            outcomes.append(not r["parse_error"] and not violations)

    for dialogue in base:
        r = run_a2_dialogue(client, dialogue, run_id=run_id + "_c")
        outcomes.append(not a2_clean_intervened(r, dialogue))

    return outcomes


def _load_records(path: Path) -> List[dict]:
    """加载 JSON 数组或 JSONL 记录列表。"""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def _load_inputs(pub: Path, args):
    """解析评测输入，返回 (base_path, defects_path, base, defects)；失败返回 None。

    默认公开轨。只给 --input 不给 --defects 时进入观测模式：数据没有缺陷标签，
    只测干净样本上的干预率等无标签指标，不产出 recall。
    """
    base_path = Path(args.input) if args.input else pub / "base_v1.json"
    if not base_path.is_file():
        print(f"[ERROR] 输入文件不存在: {base_path}", file=sys.stderr)
        return None
    base = _load_records(base_path)

    if args.defects:
        defects_path: Optional[Path] = Path(args.defects)
        if not defects_path.is_file():
            print(f"[ERROR] 缺陷文件不存在: {defects_path}", file=sys.stderr)
            return None
        defects = _load_records(defects_path)
    elif args.input:
        defects_path = None
        defects = []
    else:
        defects_path = pub / "defects_v1.jsonl"
        defects = _load_records(defects_path)

    if args.limit_base:
        if defects:
            print("[ERROR] --limit-base 仅用于观测模式（无缺陷标签）", file=sys.stderr)
            return None
        base = base[: args.limit_base]

    if defects_path is None:
        print(
            f"[INFO] 观测模式：仅跑 {len(base)} 条样本，不产出检出率/召回",
            file=sys.stderr,
        )

    return base_path, defects_path, base, defects


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.agent_eval")
    ap.add_argument("--arm", choices=list(ARMS), help="单臂评测")
    ap.add_argument("--compare", default="",
                    help="L3 跨臂对比：\"A3,A2\" 或 \"A4,A3L\" 格式")
    ap.add_argument("--live", action="store_true", help="在线臂：需要 .env 中的密钥")
    ap.add_argument("--input", default="",
                    help="干净对话 JSON 路径；默认 agent/fixtures/public/base_v1.json")
    ap.add_argument("--defects", default="",
                    help="缺陷记录 JSONL 路径；指定 --input 时可省略（观测模式，无标签）")
    ap.add_argument("--limit-base", type=int, default=0,
                    help="最多取 N 条干净样本；仅观测模式生效（调试用）")
    ap.add_argument("--limit", type=int, default=0, help="每类最多取 N 条（调试用）")
    ap.add_argument("--metric", default="blocked", choices=["blocked", "detected", "clean", "no_leak"],
                    help="L3 对比用的 outcome 定义，默认 blocked（最严格门禁）")
    ap.add_argument("--tie-rule", default="exclude", choices=["exclude", "count_half"],
                    help="胜率平局处理：exclude 排除出分母（默认）或 count_half 各计 0.5")
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
    loaded = _load_inputs(pub, args)
    if loaded is None:
        return 2
    base_path, defects_path, base, defects = loaded
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
        # 每次运行独立账本，避免多次尝试混进同一文件
        stamp_for_ledger = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        ledger_dir = root / "logs" / "agent"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        tag = args.compare.replace(",", "_") if args.compare else args.arm
        ledger = TokenLedger(ledger_dir / f"ledger_{tag}_{stamp_for_ledger}.jsonl")
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
        parts = [s.strip() for s in args.compare.split(",") if s.strip()]
        if len(parts) != 2:
            print("[ERROR] --compare 必须是恰好两个臂，如 \"A3,A2\"", file=sys.stderr)
            return 2
        arm_a, arm_b = parts
        for arm in (arm_a, arm_b):
            if arm not in ARMS:
                print(f"[ERROR] 未知臂: {arm}", file=sys.stderr)
                return 2

        payload = compare_arms(arm_a, arm_b, base, defects, client=client,
                               run_id=f"compare_{arm_a}_vs_{arm_b}_{stamp}",
                               ledger=ledger, metric=args.metric,
                               tie_rule=args.tie_rule, trace=trace)
        uses_judge = args.metric == "no_leak" or any(
            arm in (A4, A4R) for arm in (arm_a, arm_b))
        payload["run"] = {
            "limit": args.limit,
            "limit_base": args.limit_base,
            "data_mode": "observational" if defects_path is None else "defects",
            "n_defects": len(defects),
            "n_base": len(base),
            "a2_temperature": A2_TEMPERATURE if A2 in (arm_a, arm_b) else None,
            "generated_at": stamp,
        }
        # 对比结果同样必须可追溯：指纹 + manifest，避免不同尺子被混算
        input_fp = {"base": input_fingerprint(base_path, root=root)}
        if defects_path:
            input_fp["defects"] = input_fingerprint(defects_path, root=root)
        payload["fingerprint"] = {
            "input": input_fp,
            "evaluator": evaluator_fingerprint(
                rules_version=RULES_VERSION,
                judge_prompt_version=JUDGE_VERSION,
                judge_model=judge_model_name if uses_judge else "none",
                rubric_version=RUBRIC_VERSION,
                judge_temperature=0.0,
            ),
        }
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
            write_manifest(
                build_manifest(
                    root=root,
                    profile="compare",
                    arms=[arm_a, arm_b],
                    models={"generate": model_name,
                            "judge": judge_model_name if uses_judge else None},
                    extra={"limit": args.limit, "limit_base": args.limit_base,
                           "data_mode": "observational" if defects_path is None else "defects",
                           "n_defects": len(defects),
                           "n_base": len(base), "metric": args.metric,
                           "tie_rule": args.tie_rule,
                           "a2_temperature": A2_TEMPERATURE if A2 in (arm_a, arm_b) else None,
                           "compare": [arm_a, arm_b]},
                ),
                out.parent / f"manifest_{arm_a}_vs_{arm_b}.json",
            )
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
    uses_judge = args.arm in (A4, A4R)
    payload["run"] = {
        "limit": args.limit,
        "limit_base": args.limit_base,
        "data_mode": "observational" if defects_path is None else "defects",
        "n_defects": len(defects),
        "n_base": len(base),
        "a2_temperature": A2_TEMPERATURE if args.arm == A2 else None,
        "generated_at": stamp,
    }
    # 结果必须自带出处：输入与评估器指纹进结果，环境信息单独写 manifest（不判等）
    input_fp = {"base": input_fingerprint(base_path, root=root)}
    if defects_path:
        input_fp["defects"] = input_fingerprint(defects_path, root=root)
    payload["fingerprint"] = {
        "input": input_fp,
        "evaluator": evaluator_fingerprint(
            rules_version=RULES_VERSION,
            judge_prompt_version=JUDGE_VERSION,
            judge_model=judge_model_name if uses_judge else "none",
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
                extra={"limit": args.limit, "limit_base": args.limit_base,
                       "data_mode": "observational" if defects_path is None else "defects",
                       "n_defects": len(defects), "n_base": len(base),
                       "a2_temperature": A2_TEMPERATURE if args.arm == A2 else None},
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
