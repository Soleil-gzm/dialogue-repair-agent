"""Planning&Executor 测试：规划必须真正改变执行范围，且不得硬编码轮次。"""

from agent.agent.react import Plan, PlanningExecutorAgent


class _StubRuntime:
    def __init__(self):
        self.runs = []

    def run(self, dialogue, reference=None):
        self.runs.append({"dialogue": dialogue, "reference": reference})
        return {"ran": True}


class _Factory:
    """记录 runtime_factory 收到的忽略集，并返回可复用的 stub runtime。"""

    def __init__(self):
        self.ignored_sets = []
        self.runtimes = []

    def __call__(self, ignored):
        self.ignored_sets.append(ignored)
        rt = _StubRuntime()
        self.runtimes.append(rt)
        return rt


def _dialogue():
    # assistant 下标：1, 3, 5, 7
    return {"messages": [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a5"},
        {"role": "user", "content": "u6"},
        {"role": "assistant", "content": "a7"},
    ]}


class TestPlanning:
    def test_plan_has_no_hardcoded_turns(self):
        """默认不排除任何轮——证明没有被写死成 (2, 4)。"""
        agent = PlanningExecutorAgent(_Factory())
        plan = agent.plan(_dialogue())
        assert plan.candidates == [1, 3, 5, 7]

    def test_plan_respects_explicit_exclude_turns(self):
        agent = PlanningExecutorAgent(_Factory(), exclude_turns=[3, 7])
        plan = agent.plan(_dialogue())
        assert plan.candidates == [1, 5]

    def test_execute_ignores_non_candidate_turns(self):
        factory = _Factory()
        agent = PlanningExecutorAgent(factory)
        plan = Plan(candidates=[1, 5])
        agent.execute(_dialogue(), plan)

        # 非候选轮（3、7）必须交给判官忽略——规划由此真正收窄了检查范围
        assert factory.ignored_sets == [frozenset({3, 7})]

    def test_run_plans_then_executes(self):
        factory = _Factory()
        agent = PlanningExecutorAgent(factory, exclude_turns=[1])
        out = agent.run(_dialogue())

        assert out == {"ran": True}
        # 候选为 [3,5,7] → 忽略集为 {1}
        assert factory.ignored_sets == [frozenset({1})]
        assert len(factory.runtimes) == 1
        assert len(factory.runtimes[0].runs) == 1

    def test_all_assistant_turns_excluded_yields_no_candidates(self):
        factory = _Factory()
        agent = PlanningExecutorAgent(factory, exclude_turns=[1, 3, 5, 7])
        out = agent.run(_dialogue())
        assert out == {"ran": True}
        # 全部被排除 → 忽略集是全部 assistant 轮
        assert factory.ignored_sets == [frozenset({1, 3, 5, 7})]
