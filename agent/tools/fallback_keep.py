"""fallback_keep 工具

兜底策略：任何校验不通过或不确定的情况，都保留原文。
宁可不变，也不能改坏——这是数据生产场景的底线。
"""

from typing import Any, Dict

from agent.runtime.registry import FunctionTool

SCHEMA = {
    "type": "object",
    "properties": {
        "original": {
            "type": "object",
            "description": "原始对话，形如 {\"messages\": [...]}",
        },
    },
    "required": ["original"],
}


def _run(state: Any, args: Dict) -> Dict:
    original = args.get("original")
    if not isinstance(original, dict):
        return {
            "ok": False,
            "terminal": True,
            "status": "failed",
            "error": "original 必须是对话对象",
        }
    return {
        "ok": True,
        "terminal": True,
        "status": "fallback",
        "dialogue": original,
        "reason": "校验未通过，保留原文",
    }


FALLBACK_KEEP = FunctionTool(
    name="fallback_keep",
    description="放弃修改，返回原始对话。用于校验失败或不确定时兜底。",
    schema=SCHEMA,
    fn=_run,
)

__all__ = ["FALLBACK_KEEP", "SCHEMA"]
