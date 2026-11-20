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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from ..canonical import dialogue_key as _dialogue_key

from .coherence import C3_OFF_TOPIC

JUDGE_VERSION = "coherence-judge-v2"
RUBRIC_VERSION = "2.0"

# 单条消息渲染上限：判官只需判断意图是否对上，不需要读完整长文
MAX_CONTENT_CHARS = 600
MAX_TURNS = 60

SYSTEM_PROMPT = (
    "你是对话数据质检员，只判断客服回复是否跑题（答非所问）。只输出 json，不要输出任何解释性文字。\n"
    "逐条 assistant 回复按以下步骤判断：\n"
    "1. 看该回复之前最近一条 user 消息，确定用户这一轮在问什么。\n"
    "2. 只要回复仍在这个话题上推进，就算回应：直接回答、承认并记录、承诺转交/升级，"
    "或给出替代方案与下一步动作（如“稍后短信通知”“已帮您登记”“正在查询”）。\n"
    "3. 只有回复明显在讲另一件事时才判 off_topic：答的是别的问题、模板串线，"
    "或内容与用户诉求没有任何语义关联。\n"
    "4. 不要因为回答不具体、不完整、没给出确切时间/型号/金额而判 off_topic。\n"
    "5. 拿不准时不要报问题。\n"
    "示例（均不应判 off_topic）：\n"
    "user: 这个型号和页面写的不一样 -> assistant: 您反馈的问题我记录下来了，会转给对应同事。\n"
    "user: 能不能给个大概时间 -> assistant: 稍后会有短信通知，请留意。\n"
    "user: 帮我催一下 -> assistant: 我已为您催办，预计两个工作日内有进展。"
)

USER_TEMPLATE = (
    "对话如下（方括号内为消息下标 turn）：\n"
    "{rendered}\n\n"
    '请输出 json：{{"issues": [{{"turn": <assistant 消息下标>, '
    '"type": "off_topic", "detail": "<一句话，不超过 20 字>"}}]}}\n'
    '没有问题时输出 {{"issues": []}}，不要输出多余字段'
)


class JudgeError(RuntimeError):
    """判官输出不符合契约。"""


class ChatClient(Protocol):
    """判官只依赖最小聊天接口，不依赖 runtime 的具体客户端实现。"""

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        ...


def render_dialogue(dialogue: Dict[str, Any]) -> str:
    """把对话渲染成带下标的纯文本，控制单条长度以约束 token。"""
    lines = []
    for i, msg in enumerate(list((dialogue or {}).get("messages") or [])[:MAX_TURNS]):
        content = str(msg.get("content", ""))
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS] + "…"
        lines.append(f"[{i}] {msg.get('role')}: {content}")
    return "\n".join(lines)


def parse_verdict(text: str, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """解析判官输出；不合契约直接抛 JudgeError。"""
    if not str(text or "").strip():
        # 实测：推理型模型会把 max_tokens 全花在思考上，content 为空。
        # 这不是"没有违规"，必须显式失败。
        raise JudgeError("判官返回空内容（可能被 max_tokens 截断）")

    try:
        data = json.loads(text)
    except (TypeError, ValueError) as e:
        raise JudgeError(f"判官输出不是合法 json: {e}") from e

    if not isinstance(data, dict):
        raise JudgeError("判官输出必须是 json 对象")

    issues = data.get("issues", [])
    if not isinstance(issues, list):
        raise JudgeError("issues 必须是数组")

    assistant_turns = {
        i for i, m in enumerate((dialogue or {}).get("messages") or [])
        if m.get("role") == "assistant"
    }

    out: List[Dict[str, Any]] = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        turn = item.get("turn")
        # 只认 assistant 轮：判官指向 user/system 或越界时丢弃，不做修正
        if not isinstance(turn, int) or turn not in assistant_turns:
            continue
        out.append({
            "rule": C3_OFF_TOPIC,
            "turn": turn,
            "detail": str(item.get("detail") or "回复与当轮诉求无关"),
        })
    return out


@dataclass
class LLMJudge:
    """把 LLMClient 适配成 check_coherence 的判官。"""

    client: ChatClient
    version: str = JUDGE_VERSION
    # 推理型模型（如 deepseek-flash）会先消耗思考 token；给小了会导致 content 为空
    max_tokens: int = 1024
    stage: str = "judge"
    run_id: str = "adhoc"
    extra: Dict[str, Any] = field(default_factory=dict)
    # 累计用量：工具据此回传本次调用的增量，让 loop 的 token 预算覆盖判官消耗
    tokens_used: int = 0

    def __call__(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": USER_TEMPLATE.format(
                       rendered=render_dialogue(dialogue))}]

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
        for v in violations:
            v["judge_version"] = self.version
        return violations


__all__ = [
    "JUDGE_VERSION", "RUBRIC_VERSION", "JudgeError", "LLMJudge",
    "parse_verdict", "render_dialogue",
]
