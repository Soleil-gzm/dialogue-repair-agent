"""check_coherence 工具

把 eval/coherence.py 包成可注册工具：

- 默认档位只用确定性规则（重复空转 / 退化轮次），离线可复现
- 语义判官（D1 答非所问）通过 `make_check_coherence(judge=...)` 注入，
  因此工具本身不绑定任何厂商，也不在 import 时就发起网络请求
"""

from typing import Any, Dict, Optional

from agent.eval.coherence import Judge, check_coherence
from agent.runtime.registry import FunctionTool

SCHEMA = {
    "type": "object",
    "properties": {
        "dialogue": {
            "type": "object",
            "description": "待检查的对话，形如 {\"messages\": [...]}",
        },
    },
    "required": ["dialogue"],
}


def make_check_coherence(judge: Optional[Judge] = None) -> FunctionTool:
    def _run(state: Any, args: Dict) -> Dict:
        dialogue = args.get("dialogue")
        if not isinstance(dialogue, dict):
            return {
                "ok": False,
                "violations": [{"rule": "C0_input_contract",
                                "detail": "dialogue 必须是对话对象", "turn": None}],
                "tokens": 0,
            }

        # 判官内部也会调模型：把本次消耗的增量回传给 loop，才能计入 token 预算
        used_before = int(getattr(judge, "tokens_used", 0) or 0)
        violations = check_coherence(dialogue, judge=judge)
        used_after = int(getattr(judge, "tokens_used", used_before) or used_before)

        return {
            "ok": not violations,
            "violation_count": len(violations),
            "violations": violations,
            "tokens": max(0, used_after - used_before),
        }

    return FunctionTool(
        name="check_coherence",
        description=(
            "检查对话自身的连贯性：相邻 assistant 轮次是否重复空转、"
            "回复是否为空或过短；配置语义判官时还会判定答非所问。"
            "返回 ok 与违规明细。"
        ),
        schema=SCHEMA,
        fn=_run,
    )


CHECK_COHERENCE = make_check_coherence()

__all__ = ["CHECK_COHERENCE", "SCHEMA", "make_check_coherence"]
