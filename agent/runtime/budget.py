"""预算与终止条件

Agent 循环必须有硬上限，否则一次坏输入就能跑成死循环。
四类上限全部是"墙"而不是"建议"：任一触发即终止，且终止原因进 Trace。
"""

import time
from dataclasses import dataclass, field
from typing import Optional

# 终止原因
EXHAUSTED_ITERATIONS = "max_iterations"
EXHAUSTED_TOOL_CALLS = "max_tool_calls"
EXHAUSTED_TOKENS = "max_tokens"
EXHAUSTED_TIME = "max_wall_clock"


@dataclass
class Budget:
    max_iterations: int = 3
    max_tool_calls: int = 6
    max_tokens: int = 6000
    max_wall_clock_s: float = 120.0

    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def exhausted(self) -> Optional[str]:
        """返回第一个触发的上限原因；未触顶返回 None。"""
        if self.iterations >= self.max_iterations:
            return EXHAUSTED_ITERATIONS
        if self.tool_calls >= self.max_tool_calls:
            return EXHAUSTED_TOOL_CALLS
        if self.tokens >= self.max_tokens:
            return EXHAUSTED_TOKENS
        if self.elapsed_s() >= self.max_wall_clock_s:
            return EXHAUSTED_TIME
        return None

    def tick_iteration(self) -> None:
        self.iterations += 1

    def consume(self, *, tool_calls: int = 0, tokens: int = 0) -> None:
        self.tool_calls += tool_calls
        self.tokens += tokens

    def snapshot(self) -> dict:
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "elapsed_s": round(self.elapsed_s(), 3),
            "limits": {
                "max_iterations": self.max_iterations,
                "max_tool_calls": self.max_tool_calls,
                "max_tokens": self.max_tokens,
                "max_wall_clock_s": self.max_wall_clock_s,
            },
        }
