"""可注册工具集

工具必须：纯函数、无 I/O、无网络、不依赖 LLM —— 这样可脱离模型单测，
也才能在 benchmark 的离线档位里跑出确定性结果。
"""

from typing import List

from agent.runtime.registry import ToolRegistry

from .check_consistency import CHECK_CONSISTENCY
from .fallback_keep import FALLBACK_KEEP

DEFAULT_TOOLS = [CHECK_CONSISTENCY, FALLBACK_KEEP]


def register_all() -> List[str]:
    """注册全部默认工具，返回实际注册的工具名。

    幂等：已注册的工具会跳过，便于重复调用。
    """
    registered = []
    existing = set(ToolRegistry.list_tools())
    for tool in DEFAULT_TOOLS:
        if tool.name in existing:
            continue
        ToolRegistry.register(tool)
        registered.append(tool.name)
    return registered


__all__ = ["DEFAULT_TOOLS", "register_all", "CHECK_CONSISTENCY"]
