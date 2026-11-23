"""语义判官（LLM）

只做一件事：判断 assistant 回复是否**答非所问**（D1）。规则层抓不到这类问题，
字面相似度也只能命中"复制了邻轮内容"的巧合。

契约与 eval/coherence.Judge 一致：对话 -> 违规列表，可直接注入 check_coherence。

三条工程约束：
1. 输出强约束为 json，解析失败就抛错；上层按 C0_judge_error 处理 → 保守回退，
   绝不把"没验成"当成"没问题"
2. 只接受指向 assistant 轮的判定，越界或指向 user/system 的一律丢弃，
   防止判官编造位置
3. 版本号随违规返回：换 prompt 等于换尺子，报告里必须可区分
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Protocol, Tuple

from ..canonical import dialogue_key as _dialogue_key

from .coherence import C3_OFF_TOPIC

JUDGE_VERSION = "coherence-judge-v3"
RUBRIC_VERSION = "3.0"

# 单条消息渲染上限：判官只需判断意图是否对上，不需要读完整长文
MAX_CONTENT_CHARS = 600
MAX_TURNS = 60

# v3 校准（公开轨 D1 作开发集，qwen3:8b）：D1 召回 20/20，干净样本误报 0/12。
# 关键改动：按“用户诉求 -> 客服回复”成对渲染，并要求逐对输出 related 判定，
# 避免小模型只回一个空对象、把漏判伪装成“无问题”。
SYSTEM_PROMPT = (
    "你是对话数据质检员，只判断客服回复是否答非所问（off_topic）。只输出 json。\n"
    "输入按“用户诉求 -> 客服回复”成对给出，每对带 assistant 消息下标。\n"
    "对每一对都要给出 related 判断，不能遗漏：\n"
    "- 客服直接回答该问题、承认并记录、承诺转交/升级、说明正在查询、"
    "或给出与该问题相关的下一步 → related=true。\n"
    "- 客服讲的是另一个业务事项，与用户这一轮的问题没有语义关联；"
    "或只是与诉求无关的通用话术 → related=false。\n"
    "- 回复不具体、不完整、没有确切时间/型号/金额 → 仍算 related=true。\n"
    "- 语气不礼貌 → 仍算 related=true。\n"
    "判断依据是主题是否相关，不是回答质量。"
)

USER_TEMPLATE = (
    "成对数据如下：\n"
    "{pairs}\n\n"
    "必须为每一对输出一条判断，只输出 json："
    '{{"verdicts": [{{"turn": <assistant 下标>, "related": true 或 false, '
    '"detail": "<不超过 20 字>"}}]}}\n'
    "related=false 表示该回复答非所问；不要遗漏任何一对，不要输出其他字段。"
)


class JudgeError(RuntimeError):
    """判官输出不符合契约。"""


class ChatClient(Protocol):
    """判官只依赖最小聊天接口，不依赖 runtime 的具体客户端实现。"""

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        ...


def _clip(content: Any) -> str:
    text = str(content or "")
    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS] + "…"
    return text


def iter_pairs(dialogue: Dict[str, Any]) -> Iterator[Tuple[int, int, str, str]]:
    """按“user 诉求 -> 紧随其后的 assistant 回复”成对遍历。

    yields: (user_turn, assistant_turn, user_content, assistant_content)
    """
    pending: Any = None
    messages = list((dialogue or {}).get("messages") or [])[:MAX_TURNS]
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "user":
            pending = (i, _clip(msg.get("content")))
        elif role == "assistant" and pending is not None:
            yield pending[0], i, pending[1], _clip(msg.get("content"))
            pending = None


def render_pairs(dialogue: Dict[str, Any]) -> str:
    """成对渲染：降低小模型对齐“哪条回复对应哪条诉求”的难度。"""
    return "\n\n".join(
        f"user[{u}] 诉求：{uc}\nassistant[{a}] 回复：{ac}"
        for u, a, uc, ac in iter_pairs(dialogue)
    )


def render_dialogue(dialogue: Dict[str, Any]) -> str:
    """把对话渲染成带下标的纯文本，控制单条长度以约束 token。"""
    lines = []
    for i, msg in enumerate(list((dialogue or {}).get("messages") or [])[:MAX_TURNS]):
        lines.append(f"[{i}] {msg.get('role')}: {_clip(msg.get('content'))}")
    return "\n".join(lines)


# 小模型可能重复顶层键（{"verdicts": [...], "verdicts": [...]}），
# 因此不依赖整体 json 合法，而是逐对象提取 turn/related 判定。
_PAIR_TURN_FIRST = re.compile(
    r'"turn"\s*:\s*(\d+)[^{}]*?"related"\s*:\s*"?(true|false)"?'
    r'(?:[^{}]*?"detail"\s*:\s*"((?:\\.|[^"\\])*)")?',
    re.IGNORECASE,
)
_PAIR_RELATED_FIRST = re.compile(
    r'"related"\s*:\s*"?(true|false)"?[^{}]*?"turn"\s*:\s*(\d+)'
    r'(?:[^{}]*?"detail"\s*:\s*"((?:\\.|[^"\\])*)")?',
    re.IGNORECASE,
)


def _unescape(raw: Any) -> str:
    if not raw:
        return ""
    try:
        return json.loads(f'"{raw}"')
    except (TypeError, ValueError):
        return str(raw)


def _extract_verdicts(text: str) -> Dict[int, Tuple[bool, str]]:
    found: Dict[int, Tuple[bool, str]] = {}
    for m in _PAIR_TURN_FIRST.finditer(text):
        turn = int(m.group(1))
        related = m.group(2).lower() == "true"
        detail = _unescape(m.group(3))
        if turn in found and found[turn][0] != related:
            raise JudgeError(f"判官对 turn {turn} 给出冲突判定")
        found.setdefault(turn, (related, detail))
    for m in _PAIR_RELATED_FIRST.finditer(text):
        turn = int(m.group(2))
        related = m.group(1).lower() == "true"
        detail = _unescape(m.group(3))
        # 同一 turn 出现两种判法时无法采信，交给上层按判官故障处理
        if turn in found and found[turn][0] != related:
            raise JudgeError(f"判官对 turn {turn} 给出冲突判定")
        found.setdefault(turn, (related, detail))
    return found


def parse_verdict(text: str, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """解析判官输出；不合契约直接抛 JudgeError。

    v3 契约：每条成对回复都必须有 related 判定。漏判不是“没问题”，
    而是“没验成”，必须显式失败。
    """
    if not str(text or "").strip():
        # 实测：推理型模型会把 max_tokens 全花在思考上，content 为空。
        # 这不是"没有违规"，必须显式失败。
        raise JudgeError("判官返回空内容（可能被 max_tokens 截断）")

    expected = {a for _, a, _, _ in iter_pairs(dialogue)}
    if not expected:
        return []

    found = _extract_verdicts(text)
    if not found:
        raise JudgeError("判官输出缺少逐对判定（verdicts）")

    seen: Dict[int, Tuple[bool, str]] = {}
    for turn, verdict in found.items():
        # 只认成对的 assistant 轮：指向 user/system 或越界时丢弃，不做修正
        if turn not in expected:
            continue
        seen[turn] = verdict

    missing = expected - set(seen)
    if missing:
        raise JudgeError(f"判官漏判 assistant 轮: {sorted(missing)}")

    return [
        {
            "rule": C3_OFF_TOPIC,
            "turn": turn,
            "detail": detail or "回复与当轮诉求无关",
        }
        for turn, (related, detail) in sorted(seen.items())
        if not related
    ]


@dataclass
class LLMJudge:
    """把 LLMClient 适配成 check_coherence 的判官。"""

    client: ChatClient
    version: str = JUDGE_VERSION
    # 推理型模型（如 qwen3 / deepseek-flash）会先消耗思考 token；给小了会导致 content 为空
    max_tokens: int = 2048
    stage: str = "judge"
    run_id: str = "adhoc"
    extra: Dict[str, Any] = field(default_factory=dict)
    # 不参与判分的 assistant 轮（下标）。私有轨首轮开场白是业务固定话术，
    # 与随机开场句无承接关系，判官会稳定误标；在此显式排除。
    # 仍要求判官输出该轮判定（v3 契约），只是不采信。
    ignore_turns: frozenset = frozenset()
    # 累计用量：工具据此回传本次调用的增量，让 loop 的 token 预算覆盖判官消耗
    tokens_used: int = 0

    def __call__(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": USER_TEMPLATE.format(
                       pairs=render_pairs(dialogue))}]

        resp = self.client.chat(
            payload,
            temperature=0.0,
            max_tokens=self.max_tokens,
            stage=self.stage,
            run_id=self.run_id,
            dialogue_key=_dialogue_key(dialogue) if isinstance(dialogue, dict) else "",
        )
        self.tokens_used += resp.total_tokens

        violations = parse_verdict(resp.text, dialogue)
        if self.ignore_turns:
            violations = [v for v in violations
                          if v.get("turn") not in self.ignore_turns]
        for v in violations:
            v["judge_version"] = self.version
        return violations


__all__ = [
    "JUDGE_VERSION", "RUBRIC_VERSION", "JudgeError", "LLMJudge",
    "iter_pairs", "parse_verdict", "render_dialogue", "render_pairs",
]
