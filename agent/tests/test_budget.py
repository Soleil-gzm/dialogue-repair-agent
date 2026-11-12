"""测试：预算与终止条件"""

from agent.runtime.budget import (
    Budget, EXHAUSTED_ITERATIONS, EXHAUSTED_TIME, EXHAUSTED_TOKENS,
    EXHAUSTED_TOOL_CALLS,
)


class TestBudget:
    def test_not_exhausted_initially(self):
        assert Budget().exhausted() is None

    def test_iterations_limit(self):
        b = Budget(max_iterations=1)
        b.tick_iteration()
        assert b.exhausted() == EXHAUSTED_ITERATIONS

    def test_tool_calls_limit(self):
        b = Budget(max_tool_calls=2)
        b.consume(tool_calls=2)
        assert b.exhausted() == EXHAUSTED_TOOL_CALLS

    def test_tokens_limit(self):
        b = Budget(max_tokens=100)
        b.consume(tokens=100)
        assert b.exhausted() == EXHAUSTED_TOKENS

    def test_wall_clock_limit(self):
        b = Budget(max_wall_clock_s=0.0)
        assert b.exhausted() == EXHAUSTED_TIME

    def test_first_hit_wins(self):
        b = Budget(max_iterations=1, max_tool_calls=1)
        b.tick_iteration()
        b.consume(tool_calls=1)
        assert b.exhausted() == EXHAUSTED_ITERATIONS

    def test_snapshot_contains_limits(self):
        snap = Budget(max_iterations=5).snapshot()
        assert snap["limits"]["max_iterations"] == 5
        assert "elapsed_s" in snap
