"""退化指标：改写是否把数据洗坏

三条指标（Benchmark 评测设计 §4.3），只看 assistant 文本：
- 长度比：重写后字符数 / 原字符数，门槛 0.8–1.3，防"注水变长"；
- distinct-2 变化：(后 − 前) / 前，降幅 ≤ 5%，防同质化；
- 跨样本 4-gram 重复率变化，升幅 ≤ 5%，防"洗成一个味"。

为什么只取 assistant：user / system 不允许改，混进来会把"没动"算成"没退化"。

纯函数、只依赖标准库，可独立单测；口径与 report 展示共用同一份计算。
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

# §4.3 门槛
LENGTH_RATIO_MIN = 0.8
LENGTH_RATIO_MAX = 1.3
MAX_DISTINCT_DROP = 0.05
MAX_REPEAT_RISE = 0.05


def assistant_text(dialogue: Dict[str, Any]) -> str:
    """抽取一轮对话里的全部 assistant 内容。"""
    messages = (dialogue or {}).get("messages") or []
    return "\n".join(
        str(m.get("content") or "")
        for m in messages
        if m.get("role") == "assistant"
    )


def _normalized(text: Any) -> str:
    """去掉空白：中文没有词边界，字符级 n-gram 比按空白切分稳。"""
    return "".join(str(text or "").split())


def _ngrams(text: Any, n: int) -> List[str]:
    if n <= 0:
        raise ValueError(f"n 必须为正整数，收到 {n}")
    cleaned = _normalized(text)
    return [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)]


def length_ratio(before_texts: Sequence[str],
                 after_texts: Sequence[str]) -> Optional[float]:
    """语料级长度比 = 重写后总字符数 / 原总字符数。

    语料级而不是逐样本平均：避免极短样本的比值把整体拉偏。
    原文为空时无定义，返回 None（不能伪装成 1.0）。
    """
    before = sum(len(_normalized(t)) for t in before_texts)
    after = sum(len(_normalized(t)) for t in after_texts)
    if before == 0:
        return None
    return round(after / before, 6)


def distinct_n(texts: Sequence[str], n: int = 2) -> Optional[float]:
    """语料级 distinct-n = 唯一 n-gram 数 / 总 n-gram 数。"""
    grams: List[str] = []
    for text in texts:
        grams.extend(_ngrams(text, n))
    if not grams:
        return None
    return round(len(set(grams)) / len(grams), 6)


def cross_sample_repeat_rate(texts: Sequence[str], n: int = 4) -> Optional[float]:
    """跨样本重复率 = 被 ≥2 条样本共用的 n-gram 数 / 全部唯一 n-gram 数。

    样本内先按集合去重，避免一条长对话自身重复把分数抬高；
    分母取并集，衡量的是"不同对话之间撞了多少词"。
    """
    counts: Counter = Counter()
    for text in texts:
        counts.update(set(_ngrams(text, n)))
    if not counts:
        return None
    shared = sum(1 for c in counts.values() if c >= 2)
    return round(shared / len(counts), 6)


def _change(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or before == 0:
        return None
    return round((after - before) / before, 6)


def _verdict(ok: bool, value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "pass" if ok else "fail"


def measure(before_dialogues: Sequence[Dict[str, Any]],
            after_dialogues: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """对一组「输入 → 输出」对话计算三条退化指标。

    输入与输出必须一一对应；本轮没有实际执行的样本（断点续跑跳过）不要传进来，
    结果里的 n_samples 即实际用于统计的样本数。
    """
    if len(before_dialogues) != len(after_dialogues):
        raise ValueError(
            f"输入输出长度不一致: before={len(before_dialogues)}, "
            f"after={len(after_dialogues)}"
        )

    before_texts = [assistant_text(d) for d in before_dialogues]
    after_texts = [assistant_text(d) for d in after_dialogues]

    before_chars = sum(len(_normalized(t)) for t in before_texts)
    after_chars = sum(len(_normalized(t)) for t in after_texts)
    ratio = length_ratio(before_texts, after_texts)

    distinct_before = distinct_n(before_texts, 2)
    distinct_after = distinct_n(after_texts, 2)
    distinct_change = _change(distinct_before, distinct_after)

    repeat_before = cross_sample_repeat_rate(before_texts, 4)
    repeat_after = cross_sample_repeat_rate(after_texts, 4)
    repeat_change = _change(repeat_before, repeat_after)

    length_verdict = _verdict(
        ratio is not None and LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX, ratio)
    distinct_verdict = _verdict(
        distinct_change is not None and distinct_change >= -MAX_DISTINCT_DROP,
        distinct_change)
    # 跨样本重复率越低越好：升幅超阈值才算退化
    repeat_verdict = _verdict(
        repeat_change is not None and repeat_change <= MAX_REPEAT_RISE,
        repeat_change)
    # 原样本之间毫无重复、改写后开始互相撞词：比值无定义，但属于明确退化
    if repeat_change is None and repeat_before == 0 and (repeat_after or 0) > 0:
        repeat_verdict = "fail"

    return {
        "n_samples": len(before_texts),
        "length_ratio": {
            "before_chars": before_chars,
            "after_chars": after_chars,
            "ratio": ratio,
            "verdict": length_verdict,
        },
        "distinct_2": {
            "before": distinct_before,
            "after": distinct_after,
            "change": distinct_change,
            "verdict": distinct_verdict,
        },
        "cross_sample_4gram": {
            "before": repeat_before,
            "after": repeat_after,
            "change": repeat_change,
            "verdict": repeat_verdict,
        },
        "degraded": "fail" in (length_verdict, distinct_verdict, repeat_verdict),
    }


__all__ = [
    "LENGTH_RATIO_MIN", "LENGTH_RATIO_MAX", "MAX_DISTINCT_DROP", "MAX_REPEAT_RISE",
    "assistant_text", "length_ratio", "distinct_n", "cross_sample_repeat_rate",
    "measure",
]
