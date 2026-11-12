"""Agent 会话状态

显式传递，不用全局变量：同一进程可并行处理多条对话，也便于单测。
`original` 永远保留原文，供回退与 diff 使用。
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .budget import Budget

STATUS_RUNNING = "running"


def dialogue_key(dialogue: Dict[str, Any]) -> str:
    """内容哈希作稳定主键：数据没有 dialogue_id，下标跨 run 不稳定。"""
    payload = json.dumps(dialogue, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class AgentState:
    dialogue: Dict[str, Any]      # 待处理的候选版本
    original: Dict[str, Any]      # 原始参照（校验基准、回退目标）
    budget: Budget = field(default_factory=Budget)

    history: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    status: str = STATUS_RUNNING
    result_dialogue: Optional[Dict[str, Any]] = None

    @property
    def key(self) -> str:
        """以候选内容为键：同一份原始数据的不同候选应各自独立。"""
        return dialogue_key(self.dialogue)

    def called_tools(self) -> set:
        return {h.get("tool") for h in self.history}

    def record(self, tool: str, args: Dict, result: Dict) -> None:
        """记录一次工具调用。args 只存键，避免 Trace 里出现整段对话。"""
        self.history.append({
            "tool": tool,
            "arg_keys": sorted(args.keys()),
            "ok": bool(result.get("ok", True)),
            "terminal": bool(result.get("terminal", False)),
        })
