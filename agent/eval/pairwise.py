"""E2 双向盲评（成对偏好）

为什么需要它：绝对分判官（judge.py）对"哪份更好"不敏感，容易受长度、语气影响；
成对盲评只看两份输出的相对质量，且不告诉判官哪份来自实验臂。

双向的作用：同一对样本按 A/B 与 B/A 各评一次。
- 两次都指向同一臂 → 记该臂胜；
- 两次指向不同臂 → 判官受位置影响，记 tie（矛盾），并计入一致率；
- 两次都说 tie → 记 tie。

判官故障（输出不合契约）不能静默变成 tie：单独计 errors，并从一致率分母剔除。
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .judge import JudgeError, render_dialogue

PAIRWISE_VERSION = "pairwise-blind-v1"

SYSTEM_PROMPT = (
    "你是对话数据质检员，要比较同一段对话的两个修复版本。\n"
    "只判断哪个版本更好地修复了 assistant 回复答非所问、逻辑不通的问题。\n"
    "判断只看修复质量，不受版本出现顺序影响。\n"
    "两个版本一样好、或一样差时选择 tie；不要因为文字更长就选它。\n"
    "只输出 json。"
)

USER_TEMPLATE = (
    "版本 left：\n{left}\n\n"
    "版本 right：\n{right}\n\n"
    "只输出 json："
    '{{"winner": "left 或 right 或 tie", "reason": "<不超过 30 字>"}}'
)

_WINNER = re.compile(
    r'"winner"\s*:\s*"?(left|right|tie)"?', re.IGNORECASE)


@dataclass
class PairwiseJudge:
    """把 LLMClient 适配成成对盲评判官；只依赖 chat 接口。"""

    client: Any
    version: str = PAIRWISE_VERSION
    max_tokens: int = 1024
    stage: str = "pairwise"
    run_id: str = "blind"
    tokens_used: int = 0
    calls: int = 0

    def compare(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        """返回 "left" / "right" / "tie"；输出不合契约抛 JudgeError。"""
        payload = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                left=render_dialogue(left), right=render_dialogue(right))},
        ]
        resp = self.client.chat(
            payload,
            temperature=0.0,
            max_tokens=self.max_tokens,
            stage=self.stage,
            run_id=self.run_id,
        )
        self.tokens_used += getattr(resp, "total_tokens", 0) or 0
        self.calls += 1
        return parse_winner(resp.text)


def parse_winner(text: str) -> str:
    """解析成对判官输出；缺失或冲突一律抛 JudgeError。"""
    if not str(text or "").strip():
        raise JudgeError("成对判官返回空内容")

    winners = {m.group(1).lower() for m in _WINNER.finditer(text)}
    if not winners:
        raise JudgeError("成对判官输出缺少 winner")
    if len(winners) > 1:
        raise JudgeError(f"成对判官给出冲突 winner: {sorted(winners)}")
    return winners.pop()


@dataclass
class BlindResult:
    arm_a: str
    arm_b: str
    n_pairs: int
    judgments: int
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    contradictions: int = 0
    errors: int = 0

    def consistency_rate(self) -> Optional[float]:
        """两次顺序不矛盾的样本占有效样本的比例。"""
        valid = self.n_pairs - self.errors
        if valid <= 0:
            return None
        return round((valid - self.contradictions) / valid, 4)

    def tie_rate(self) -> Optional[float]:
        if not self.n_pairs:
            return None
        return round(self.ties / self.n_pairs, 4)

    def to_dict(self, win_rate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "arms": [self.arm_a, self.arm_b],
            "judge_version": PAIRWISE_VERSION,
            "n_pairs": self.n_pairs,
            "judgments": self.judgments,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "contradictions": self.contradictions,
            "errors": self.errors,
            "consistency_rate": self.consistency_rate(),
            "tie_rate": self.tie_rate(),
            "win_rate_a": win_rate,
        }


def compare_blind(
    judge: PairwiseJudge,
    pairs: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    *,
    arm_a: str,
    arm_b: str,
) -> Dict[str, Any]:
    """对每对输出做 A/B 与 B/A 两次盲评，矛盾记 tie。

    pairs: [(sample_key, dialogue_a, dialogue_b), ...]
    """
    result = BlindResult(arm_a=arm_a, arm_b=arm_b, n_pairs=len(pairs), judgments=0)

    for _key, dialogue_a, dialogue_b in pairs:
        try:
            first = judge.compare(dialogue_a, dialogue_b)
            second = judge.compare(dialogue_b, dialogue_a)
        except JudgeError:
            result.errors += 1
            continue

        result.judgments += 2
        # 第一次 left=A、right=B；第二次左右互换，映射回臂身份
        vote_first = {"left": "A", "right": "B", "tie": "tie"}[first]
        vote_second = {"right": "A", "left": "B", "tie": "tie"}[second]

        if vote_first == vote_second and vote_first != "tie":
            if vote_first == "A":
                result.wins_a += 1
            else:
                result.wins_b += 1
        elif vote_first == "tie" and vote_second == "tie":
            result.ties += 1
        else:
            # 两次指向不同臂：位置偏差导致结论不可信，按平局计
            result.contradictions += 1
            result.ties += 1

    from ..benchmark.l3_stats import wilson_ci
    decided = result.wins_a + result.wins_b
    if decided == 0:
        win_rate = {
            "point": None, "ci_lower": None, "ci_upper": None,
            "comparable": 0, "note": "没有可判胜负的样本（全为平局/矛盾），胜率不可比",
        }
    else:
        point, lo, hi = wilson_ci(result.wins_a, decided)
        win_rate = {
            "point": point, "ci_lower": lo, "ci_upper": hi,
            "comparable": decided,
            "note": f"双向一致后才计入胜负；矛盾 {result.contradictions} 条按平局处理",
        }
    return result.to_dict(win_rate)


__all__ = [
    "PAIRWISE_VERSION", "PairwiseJudge", "BlindResult", "JudgeError",
    "compare_blind", "parse_winner",
]
