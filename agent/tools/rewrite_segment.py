"""rewrite_segment 工具

让模型只重写一条 assistant 回复，而不是复述整段对话：
策略层负责「定位 + 下指令」，模型只做局部改写。

该工具是默认工具集中唯一需要外部客户端的工具，因此不进入 DEFAULT_TOOLS，
由评测臂按需注册（离线测试注入 FakeLLMClient 即可）。
"""

import copy
import json
from typing import Any, Dict, Optional

from agent.runtime.registry import FunctionTool

REWRITE_SEGMENT = "rewrite_segment"

SCHEMA = {
    "type": "object",
    "properties": {
        "dialogue": {
            "type": "object",
            "description": "待修复的对话，形如 {\"messages\": [...]}",
        },
        "turn": {
            "type": "integer",
            "description": "要重写的 assistant 消息下标",
        },
        "issues": {
            "type": "array",
            "items": {"type": "object"},
            "description": "该轮需要修复的违规明细（rule / detail / turn）",
        },
    },
    "required": ["dialogue", "turn", "issues"],
}

SYSTEM_PROMPT = (
    "你是对话数据修复器，只改写指定的那一轮 assistant 回复。\n"
    "要求：\n"
    "1. 保持原意与已确认的事实，不新增承诺，不改变订单号、金额、时间等实体。\n"
    "2. 只修复给定的问题，保持客服口吻，长度与原句接近。\n"
    "3. 只输出 json，格式为 {\"content\": \"改写后的回复\"}，不要输出其它字段或解释。"
)


def _violation(rule: str, detail: str, turn: Any) -> Dict[str, Any]:
    return {"rule": rule, "detail": detail, "turn": turn}


def _fail(rule: str, detail: str, turn: Any, tokens: int = 0) -> Dict[str, Any]:
    return {
        "ok": False,
        "violation_count": 1,
        "violations": [_violation(rule, detail, turn)],
        "tokens": tokens,
    }


def _strip_fences(text: str) -> str:
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_content(text: str) -> Optional[str]:
    try:
        data = json.loads(_strip_fences(text))
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    content = data.get("content")
    if not isinstance(content, str):
        return None
    return content.strip()


def _system_text(dialogue: Dict[str, Any]) -> str:
    for msg in dialogue.get("messages") or []:
        if msg.get("role") == "system":
            return str(msg.get("content") or "")
    return ""


def _build_messages(dialogue: Dict[str, Any], turn: int,
                    issues: Any) -> list:
    msgs = dialogue.get("messages") or []
    target = msgs[turn]
    lines = []
    system_text = _system_text(dialogue)
    if system_text:
        lines.append("系统设定（只读，不要复述）：\n" + system_text[:800])
    if turn > 0 and msgs[turn - 1].get("role") == "user":
        lines.append("客户上一句：\n" + str(msgs[turn - 1].get("content") or ""))
    lines.append("待改写的 assistant 回复：\n" + str(target.get("content") or ""))
    if issues:
        lines.append("需要修复的问题：")
        for issue in list(issues)[:8]:
            if not isinstance(issue, dict):
                continue
            lines.append(f"- {issue.get('rule')}: {issue.get('detail')}")
    lines.append("请输出 json：{\"content\": \"改写后的回复\"}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(lines)},
    ]


def make_rewrite_segment(client: Any, *, temperature: float = 0.3,
                         max_tokens: int = 512,
                         run_id: str = "adhoc") -> FunctionTool:
    """构造带模型客户端的 rewrite_segment 工具。"""

    def _run(state: Any, args: Dict) -> Dict:
        dialogue = args.get("dialogue")
        turn = args.get("turn")
        issues = args.get("issues") or []

        if not isinstance(dialogue, dict) or not isinstance(dialogue.get("messages"), list):
            return _fail("W0_invalid_turn", "dialogue 必须是含 messages 的对话对象", turn)
        msgs = dialogue["messages"]
        if isinstance(turn, bool) or not isinstance(turn, int) or not (0 <= turn < len(msgs)):
            return _fail("W0_invalid_turn", f"非法轮次下标: {turn!r}", turn)
        target = msgs[turn] if isinstance(msgs[turn], dict) else {}
        if target.get("role") != "assistant":
            return _fail("W0_invalid_turn", f"第 {turn} 轮不是 assistant", turn)

        original = str(target.get("content") or "")
        dialogue_key = getattr(state, "key", "") or ""
        try:
            resp = client.chat(
                _build_messages(dialogue, turn, issues),
                temperature=temperature,
                max_tokens=max_tokens,
                stage="generate",
                run_id=run_id,
                dialogue_key=dialogue_key,
            )
        except Exception as e:  # 模型失败不应让循环直接 failed
            return _fail("W4_llm_error", f"模型调用失败: {type(e).__name__}", turn)

        tokens = int(getattr(resp, "total_tokens", 0) or 0)
        content = _parse_content(getattr(resp, "text", ""))
        if content is None:
            return _fail("W1_parse_error", "模型输出不是合法的 json 对象", turn, tokens)
        if not content:
            return _fail("W2_empty_content", "重写结果为空", turn, tokens)
        if content == original.strip():
            return _fail("W3_unchanged", "重写结果与原回复相同", turn, tokens)

        rewritten = copy.deepcopy(dialogue)
        rewritten["messages"][turn]["content"] = content
        return {
            "ok": True,
            "status": "rewritten",
            "dialogue": rewritten,
            "rewritten_turn": turn,
            "violation_count": 0,
            "violations": [],
            "tokens": tokens,
        }

    return FunctionTool(
        name=REWRITE_SEGMENT,
        description=(
            "重写指定 assistant 轮次以修复已定位的问题。"
            "只改该轮回复，保持角色、结构与其它轮次不变。"
        ),
        schema=SCHEMA,
        fn=_run,
    )


__all__ = ["REWRITE_SEGMENT", "SCHEMA", "make_rewrite_segment"]
