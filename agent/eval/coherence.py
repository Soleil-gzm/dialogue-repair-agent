"""连贯性检查

两类问题，两种手段：

- **D5 重复空转 / 退化轮次**：结构 + 字面相似度即可判定，纯确定性，可离线复现。
- **D1 答非所问**：必须理解语义，交给可注入的 `judge`（LLM 判官）。
  判官不在本模块实现，避免把"网络依赖"藏进评估器；无法调用时按违规处理，
  宁可回退原文，也不放行未经验证的候选。

与 eval/rules.py 的分工：rules 管"改没改坏"（保真），coherence 管"本来就不连贯"。
"""

import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

C0_INPUT_CONTRACT = "C0_input_contract"
C0_JUDGE_ERROR = "C0_judge_error"
C1_REPETITION = "C1_assistant_repetition"
C2_DEGENERATE_TURN = "C2_degenerate_turn"
C3_OFF_TOPIC = "C3_off_topic"

# 相似度阈值：只用来抓"近乎逐字复制"，避免把"同一话题换个说法"误判成重复
REPEAT_SIMILARITY = 0.9
# 低于该字数的 assistant 轮次视为空转（空串 / "嗯" / "好的"）
MIN_ASSISTANT_CHARS = 2

_PUNCT = re.compile(
    "[\\s，。！？、；：\u201c\u201d\u2018\u2019（）()【】\\[\\]…—\\-,.!?;:'\"]+"
)

# 判官契约：给定对话，返回违规列表（每条含 rule / turn / detail）
Judge = Callable[[Dict[str, Any]], List[Dict[str, Any]]]


def normalize_text(text: str) -> str:
    """去掉空白与标点并统一小写：只比内容，不比排版。"""
    return _PUNCT.sub("", str(text or "")).lower()


def similarity(a: str, b: str) -> float:
    """字符级相似度（0~1）。短文本上比对分词的词表依赖更适合开源分发。"""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _violation(rule: str, detail: str, turn: Optional[int] = None) -> Dict[str, Any]:
    return {"rule": rule, "detail": detail, "turn": turn}


def _assistant_indices(messages: List[Dict[str, Any]]) -> List[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


# --------------------------------------------------------------------------
# 确定性检查
# --------------------------------------------------------------------------

def check_repetition(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """相邻 assistant 轮次近乎逐字重复 → 重复空转。

    只比"相邻"，不做全局去重：长对话里同一句套话非相邻出现是正常现象，
    全局比对会把干净样本误判成缺陷。
    """
    msgs = list((dialogue or {}).get("messages") or [])
    idx = _assistant_indices(msgs)
    out: List[Dict[str, Any]] = []
    for prev, cur in zip(idx, idx[1:]):
        a = normalize_text(msgs[prev].get("content", ""))
        b = normalize_text(msgs[cur].get("content", ""))
        sim = similarity(a, b)
        if a and b and sim >= REPEAT_SIMILARITY:
            out.append(_violation(
                C1_REPETITION,
                f"第 {cur} 轮复读了第 {prev} 轮的回复（相似度 {sim:.2f}）",
                turn=cur,
            ))
    return out


def check_degenerate(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """assistant 轮次为空或过短 → 无效回复。"""
    msgs = list((dialogue or {}).get("messages") or [])
    out: List[Dict[str, Any]] = []
    for i in _assistant_indices(msgs):
        text = normalize_text(msgs[i].get("content", ""))
        if len(text) < MIN_ASSISTANT_CHARS:
            out.append(_violation(
                C2_DEGENERATE_TURN,
                f"第 {i} 轮 assistant 回复为空或过短（{len(text)} 字）",
                turn=i,
            ))
    return out


def check_coherence(
    dialogue: Dict[str, Any],
    *,
    judge: Optional[Judge] = None,
) -> List[Dict[str, Any]]:
    """返回连贯性违规列表；空列表表示通过。

    Args:
        dialogue: 待检查对话，形如 {"messages": [...]}
        judge: 可选的语义判官（如 LLM）。抛异常时记为 C0_judge_error，
               让调用方走保守回退，而不是把"没验成"当成"没问题"。
    """
    if not isinstance(dialogue, dict):
        return [_violation(C0_INPUT_CONTRACT, "dialogue 必须是对话对象")]

    violations = check_repetition(dialogue) + check_degenerate(dialogue)

    if judge is not None:
        try:
            extra = judge(dialogue) or []
        except Exception as e:  # 判官故障不得静默放行
            extra = [_violation(C0_JUDGE_ERROR, f"语义判官不可用: {e}")]
        violations.extend(v for v in extra if isinstance(v, dict))

    return _dedupe(violations)


def _dedupe(violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同一轮同一规则只保留一条，避免判官与规则重复计数。"""
    seen = set()
    out: List[Dict[str, Any]] = []
    for v in violations:
        key = (v.get("rule"), v.get("turn"))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out
