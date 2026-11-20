"""测试：Agent 臂评测聚合逻辑（全离线）

只验证"指标怎么算"，不验证模型好坏；真实数值由 benchmark 运行产出。
"""

import json
import random

import pytest

from agent.benchmark.agent_eval import (
    A3, A3L, A3R, A4, A4R, ArmResult, _parse_a2_output, build_runtime, run_arm,
)
from agent.benchmark.defects import generate, inject
from agent.runtime.llm import FakeLLMClient, LLMResponse
from agent.runtime.trace import TraceWriter


class _ScriptClient:
    """按脚本返回决策/工具调用，用于验证带判官与改写的完整闭环。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        return LLMResponse(
            text=item.get("text", ""),
            input_tokens=item.get("tokens", 0),
            tool_calls=item.get("tool_calls", []),
        )


def _call(name):
    return {"tool_calls": [{"id": "c1", "name": name, "arguments": {}}]}


class _CyclicClient:
    """交替返回两个校验工具调用，永不主动结束（用预算收口）。"""

    def __init__(self, tokens: int = 10):
        self.tokens = tokens
        self._i = 0

    def chat(self, messages, **kwargs):
        name = ("check_consistency", "check_coherence")[self._i % 2]
        self._i += 1
        return LLMResponse(
            text="",
            input_tokens=self.tokens,
            tool_calls=[{"name": name, "arguments": {}}],
        )


def _flag_all(_dialogue):
    return [{"rule": "C3_off_topic", "detail": "跑题", "turn": 1}]


@pytest.fixture
def dataset(mini_dialogues):
    defects = generate(mini_dialogues, per_type=2, seed=7)
    return mini_dialogues, defects


class TestBuildRuntime:
    def test_rule_arm_needs_no_client(self):
        assert build_runtime(A3) is not None

    def test_llm_arms_require_client(self):
        with pytest.raises(ValueError):
            build_runtime(A3L)
        with pytest.raises(ValueError):
            build_runtime(A3R)
        with pytest.raises(ValueError):
            build_runtime(A4)
        with pytest.raises(ValueError):
            build_runtime(A4R)

    def test_a4r_uses_rewrite_aware_policy(self):
        """A4R = A4 + rewrite_segment：决策器要允许改写，工具表里要有改写工具。"""
        from agent.runtime.registry import ToolRegistry

        runtime = build_runtime(A4R, client=FakeLLMClient())
        assert runtime.policy.allow_rewrite is True
        assert "rewrite_segment" in ToolRegistry.list_tools()

    def test_unknown_arm(self):
        with pytest.raises(ValueError):
            build_runtime("A9")


class TestTrace:
    def test_run_arm_persists_decision_trace(self, dataset, tmp_path):
        """失败对话必须能留下原因，否则 failed 计数无法归因。"""
        base, defects = dataset
        path = tmp_path / "trace.jsonl"
        run_arm(A3, base, defects[:1], trace=TraceWriter(path))

        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert any(e["event"] == "start" for e in events)
        assert any(e["event"] == "end" for e in events)


class TestRuleArm:
    def test_run_arm_tags_client_with_arm(self, dataset):
        """账本按臂归集：跑哪个臂，客户端就要被标记成哪个臂。"""
        base, defects = dataset

        class _Tagged(_CyclicClient):
            arm = "adhoc"

            def set_arm(self, arm):
                self.arm = arm

        client = _Tagged()
        run_arm(A3L, base, defects[:1], client=client)
        assert client.arm == A3L

    def test_counts_and_block_rate(self, dataset):
        base, defects = dataset
        result = run_arm(A3, base, defects)

        assert result.arm == A3
        assert set(result.per_type) == {d["defect"]["type"] for d in defects}
        for row in result.per_type.values():
            assert row["total"] == 2

        # 结构类缺陷规则层必拦
        assert result.recall("D3_fact_conflict") == 1.0
        assert result.recall("D6_missing_response") == 1.0

    def test_clean_samples_not_flagged(self, dataset):
        base, defects = dataset
        result = run_arm(A3, base, defects)
        assert result.clean_total == len(base)
        assert result.clean_flagged == 0
        assert result.clean_intervention_rate() == 0.0

    def test_rule_arm_spends_no_tokens(self, dataset):
        base, defects = dataset
        result = run_arm(A3, base, defects)
        assert result.tokens == 0
        assert result.cost_cny == 0.0


class TestLLMArmMetrics:
    def test_detected_and_blocked_are_distinct(self, dataset):
        """判官报违规但策略没回退 → detected 有、blocked 无。"""
        base, defects = dataset
        result = run_arm(
            A4, base, defects[:3],
            client=_CyclicClient(), judge=_flag_all,
        )
        for row in result.per_type.values():
            assert row["detected"] == row["total"], "判官每次都报违规"
            assert row["blocked"] == 0, "策略没调用 fallback_keep"
        assert result.clean_flagged == 0

    def test_stopped_counts_budget_terminations(self, dataset):
        """预算中止单独计数：环境中断不能被误读成"没检出"。"""
        base, defects = dataset
        # 单次决策就超 token 预算 → 每条样本都在下一轮被预算拦住
        result = run_arm(A4, base, defects[:3],
                         client=_CyclicClient(tokens=10_000), judge=_flag_all)
        assert result.stopped == len(defects[:3]) + len(base)
        assert result.failed == 0

    def test_tokens_are_accumulated(self, dataset):
        base, defects = dataset
        result = run_arm(A4, base, defects[:2], client=_CyclicClient(), judge=_flag_all)
        assert result.tokens > 0
        assert result.calls == 0, "没有 ledger 时不报调用次数"

    def test_policy_arm_without_judge(self, dataset):
        base, defects = dataset
        result = run_arm(A3L, base, defects[:2], client=_CyclicClient())
        assert result.arm == A3L
        assert result.tokens > 0
        assert all(row["detected"] == 0 for row in result.per_type.values())


class TestRewriteArm:
    def test_clean_samples_skip_llm(self, dataset):
        base, _ = dataset
        client = FakeLLMClient(default="not-json")
        result = run_arm(A3R, base, [], client=client)

        assert result.clean_total == len(base)
        assert result.clean_flagged == 0
        assert client.call_count == 0
        assert result.tokens == 0

    def test_failed_rewrite_falls_back_to_block(self, dataset):
        base, defects = dataset
        result = run_arm(A3R, base, defects, client=FakeLLMClient(default="not-json"))

        # 结构类缺陷若可重写则重写失败回退，若不可定位则直接回退，两者都算挡住
        assert result.recall("D3_fact_conflict") == 1.0
        assert result.recall("D6_missing_response") == 1.0
        assert result.failed == 0


class TestA4RRewriteClosedLoop:
    def test_rewrite_success_counts_as_repaired(self, dataset):
        """判官报 D1 → 改写 → 复检通过 → accepted，这才是 repaired 而不是 blocked。"""
        base, _ = dataset
        clean = base[0]
        bad, _ = inject(clean, "D1_off_topic", random.Random(3))
        record = {"dialogue": bad, "defect": {"type": "D1_off_topic"}, "base_index": 0}

        turn = next(
            i for i, (a, b) in enumerate(zip(clean["messages"], bad["messages"]))
            if a.get("content") != b.get("content")
        )
        fixed = clean["messages"][turn]["content"]
        off_topic = [{"rule": "C3_off_topic", "detail": "答非所问", "turn": turn}]

        calls = {"judge": 0}

        def judge(_dialogue):
            calls["judge"] += 1
            return off_topic if calls["judge"] == 1 else []

        client = _ScriptClient([
            # 缺陷样本：两次校验 → 改写 → 复检 → 结束
            _call("check_consistency"),
            _call("check_coherence"),
            _call("rewrite_segment"),
            {"text": json.dumps({"content": fixed}, ensure_ascii=False)},
            _call("check_consistency"),
            _call("check_coherence"),
            {"text": "复检通过"},
            # 干净样本：校验通过直接结束
            _call("check_consistency"),
            _call("check_coherence"),
            {"text": "校验通过"},
        ])
        result = run_arm(A4R, [clean], [record], client=client, judge=judge)

        assert result.repaired == 1
        assert result.repair_rate() == 1.0
        assert result.per_type["D1_off_topic"]["blocked"] == 0
        assert result.clean_flagged == 0
        # repaired 样本要留原始对照，供 --review 导出人工抽检
        assert len(result.samples) == 1
        assert result.samples[0]["type"] == "D1_off_topic"
        assert result.samples[0]["key"]


class TestA2Parsing:
    """A2 只采纳 assistant content：模型丢标注字段或动不可改消息都不算改坏。"""

    ORIGINAL = {
        "messages": [
            {"role": "system", "content": "角色说明\n"},
            {"role": "user", "content": "订单什么时候到？"},
            {"role": "assistant", "content": "旧回复", "loss": "True"},
        ]
    }

    def test_keeps_original_system_user_and_metadata(self):
        model_out = (
            '{"messages": ['
            '{"role": "system", "content": "角色说明"},'   # 丢了末尾换行
            '{"role": "user", "content": "订单啥时候到？"},'  # 擅自改写用户话
            '{"role": "assistant", "content": "新回复"}'      # 丢了 loss 字段
            "]}"
        )
        parsed = _parse_a2_output(model_out, self.ORIGINAL)
        assert parsed["ok"] is True
        msgs = parsed["dialogue"]["messages"]
        assert msgs[0] == self.ORIGINAL["messages"][0]
        assert msgs[1] == self.ORIGINAL["messages"][1]
        assert msgs[2]["content"] == "新回复"
        assert msgs[2]["loss"] == "True"

    def test_role_mismatch_is_parse_error(self):
        model_out = '{"messages": [{"role": "system", "content": "x"}, {"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}]}'
        assert _parse_a2_output(model_out, self.ORIGINAL)["ok"] is False

    def test_length_mismatch_is_parse_error(self):
        model_out = '{"messages": [{"role": "system", "content": "x"}]}'
        assert _parse_a2_output(model_out, self.ORIGINAL)["ok"] is False

    def test_markdown_fenced_output_still_parses(self):
        model_out = (
            "```json\n"
            '{"messages": [{"role": "system", "content": "x"}, '
            '{"role": "user", "content": "y"}, '
            '{"role": "assistant", "content": "ok"}]}\n'
            "```"
        )
        assert _parse_a2_output(model_out, self.ORIGINAL)["ok"] is True


class TestSerialization:
    def test_to_dict_fields(self):
        r = ArmResult(arm=A3, per_type={"D1_off_topic": {"total": 20, "detected": 0, "blocked": 20}},
                      clean_total=12, clean_flagged=0, tokens=100, cost_cny=0.25)
        d = r.to_dict()
        assert d["arm"] == A3
        assert d["macro_recall_blocked"] == 1.0
        assert d["clean_intervention_rate"] == 0.0
        assert d["per_type"]["D1_off_topic"]["blocked"] == 20

    def test_repair_rate_counts_only_detected(self):
        """修复成功率分母是被标记数，不是样本总数（设计 L3）。"""
        r = ArmResult(arm=A3R, repaired=2)
        r.per_type = {"D4": {"total": 3, "detected": 2, "blocked": 1},
                      "D5": {"total": 3, "detected": 2, "blocked": 1}}
        assert r.repair_rate() == pytest.approx(0.5)

    def test_clean_failed_counts_as_intervention(self):
        """干净样本未跑完属于被干预，不能美化成未误报（R7）。"""
        r = ArmResult(arm=A3, clean_total=12, clean_flagged=0, clean_failed=4)
        assert r.clean_intervention_rate() == pytest.approx(0.3333, abs=1e-4)
        assert r.to_dict()["clean_failed"] == 4

    def test_macro_recall_averages_types(self):
        r = ArmResult(arm=A3, per_type={
            "D1": {"total": 10, "detected": 10, "blocked": 10},
            "D2": {"total": 10, "detected": 5, "blocked": 0},
        })
        assert r.macro_recall("blocked") == 0.5
        assert r.macro_recall("detected") == 0.75

    def test_empty_result_is_zero_not_crash(self):
        r = ArmResult(arm=A3)
        assert r.macro_recall() == 0.0
        assert r.clean_intervention_rate() == 0.0
