"""check_consistency 工具

把 eval/rules.py 的四类硬约束包成可注册工具，供 Agent 循环调用，
也供 Benchmark 直接评估（不需要 LLM）。
"""

from typing import Any, Dict

from agent.eval.rules import check_consistency, is_consistent
from agent.runtime.registry import FunctionTool

SCHEMA = {
    "type": "object",
    "properties": {
        "before": {
            "type": "object",
            "description": "原始对话，形如 {\"messages\": [...]}",
        },
        "after": {
            "type": "object",
            "description": "待校验的对话（通常是重写结果）",
        },
    },
    "required": ["before", "after"],
}


def _run(state: Any, args: Dict) -> Dict:
    before = args.get("before")
    after = args.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {
            "ok": False,
            "violations": [{"rule": "R0_input_contract",
                            "detail": "before/after 必须是对话对象", "turn": None}],
        }

    violations = check_consistency(before, after)
    return {
        "ok": not violations,
        "violation_count": len(violations),
        "violations": violations,
    }


CHECK_CONSISTENCY = FunctionTool(
    name="check_consistency",
    description=(
        "校验重写后的对话是否保持 system 逐字不变、结构（消息数/role/loss）不变、"
        "实体完全保真、说话人集合不变。返回 ok 与违规明细。"
    ),
    schema=SCHEMA,
    fn=_run,
)

__all__ = ["CHECK_CONSISTENCY", "SCHEMA", "is_consistent"]
