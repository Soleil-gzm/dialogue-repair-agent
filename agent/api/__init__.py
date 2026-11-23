"""agent.api — FastAPI 服务化骨架。

本包把 dialogue-repair-agent 的决策循环暴露成 HTTP 端点。
核心调用链不变：build_runtime → runtime.run → AgentResult。
"""

from .server import app

__all__ = ["app"]
