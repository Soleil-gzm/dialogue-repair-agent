"""测试：Agent 臂评测聚合逻辑（全离线）

只验证"指标怎么算"，不验证模型好坏；真实数值由 benchmark 运行产出。
"""

import copy
import json
import random

import pytest

from agent.benchmark.agent_eval import (
    A2, A3, A3L, A3R, A4, A4R, ArmResult, _collect_outcomes_a2, _parse_a2_output,
    a2_clean_intervened, build_runtime, main, run_arm,
)
from agent.benchmark.defects import generate, inject, to_jsonl
from agent.benchmark.fixtures import generate_dialogues
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


class TestL3Stats:
    """统计层必须是可信的：正态 CDF 与精确二项都不能再产生假显著。"""

    def test_normal_cdf_matches_known_value(self):
        from agent.benchmark.l3_stats import _normal_cdf

        assert _normal_cdf(0.0) == pytest.approx(0.5)
        assert _normal_cdf(1.959964) == pytest.approx(0.975, abs=1e-5)
        assert _normal_cdf(-1.959964) == pytest.approx(0.025, abs=1e-5)

    def test_mcnemar_exact_4_vs_0(self):
        from agent.benchmark.l3_stats import mcnemar

        mn = mcnemar([False] * 4, [True] * 4)
        assert mn.method == "exact"
        assert mn.n10 + mn.n01 == 4
        assert mn.p_value == pytest.approx(0.125)

    def test_mcnemar_exact_2_vs_0(self):
        from agent.benchmark.l3_stats import mcnemar

        mn = mcnemar([False] * 2, [True] * 2)
        assert mn.p_value == pytest.approx(0.5)

    def test_large_sample_uses_chi2(self):
        from agent.benchmark.l3_stats import mcnemar

        # 30 个不一致对（>=25）才走卡方近似
        a = [False] * 30 + [True] * 30
        b = [True] * 30 + [True] * 30
        mn = mcnemar(a, b)
        assert mn.method == "chi2"
        assert 0.0 <= mn.p_value <= 1.0
        assert mn.p_value < 0.05

    def test_win_rate_all_ties_is_not_zero(self):
        """全平局时胜率不可比：不能返回 0.0 伪装成'A 全输'。"""
        from agent.benchmark.l3_stats import win_rate

        wr = win_rate([True, False, True], [True, False, True])
        assert wr.ties == 3
        assert wr.n_comparable == 0
        assert wr.point is None
        assert wr.ci_lower is None and wr.ci_upper is None

    def test_win_rate_exclude_uses_decided_denominator(self):
        from agent.benchmark.l3_stats import win_rate

        wr = win_rate([True, False, True], [False, False, True])
        assert wr.wins_a == 1 and wr.wins_b == 0 and wr.ties == 2
        assert wr.n_comparable == 1
        assert wr.point == 1.0

    def test_win_rate_count_half_uses_all_samples(self):
        from agent.benchmark.l3_stats import win_rate

        wr = win_rate([True, False, True], [False, False, True], tie_rule="count_half")
        assert wr.n_comparable == 3
        # A 胜 1、平 2 → (1 + 1) / 3
        assert wr.point == pytest.approx(2 / 3, abs=1e-6)

    def test_win_rate_rejects_unknown_tie_rule(self):
        from agent.benchmark.l3_stats import win_rate

        with pytest.raises(ValueError):
            win_rate([True], [True], tie_rule="nope")


class _EchoA2Client:
    """A2 桩：把输入的对话原样（或按 mutate 改写后）回显。"""

    def __init__(self, mutate=None):
        self.mutate = mutate
        self.calls = 0
        self.arm = None

    def set_arm(self, arm):
        self.arm = arm

    def chat(self, messages, **kwargs):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        if self.mutate:
            payload = self.mutate(payload)
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False),
                           input_tokens=1)


_CLEAN_DIALOGUE = {
    "messages": [
        {"role": "system", "content": "角色说明"},
        {"role": "user", "content": "订单什么时候到？"},
        {"role": "assistant", "content": "预计明天送达。"},
    ]
}
_BAD_DIALOGUE = {
    "messages": [
        {"role": "system", "content": "角色说明"},
        {"role": "user", "content": "订单什么时候到？"},
        {"role": "assistant", "content": "请先付款。"},
    ]
}
_D1_RECORD = {
    "dialogue": _BAD_DIALOGUE,
    "defect": {"type": "D1_off_topic"},
    "base_index": 0,
}


class TestA2Outcomes:
    """A2 的 outcome 必须按 metric 分流，不能所有指标都拿一致性凑数。"""

    def test_temperature_comes_from_named_constant(self):
        from agent.benchmark.agent_eval import A2_TEMPERATURE, run_a2_dialogue

        class _Capture:
            def chat(self, messages, **kwargs):
                self.kwargs = kwargs
                return LLMResponse(
                    text=json.dumps(_CLEAN_DIALOGUE, ensure_ascii=False),
                    input_tokens=1,
                )

        client = _Capture()
        run_a2_dialogue(client, _CLEAN_DIALOGUE)
        assert client.kwargs["temperature"] == A2_TEMPERATURE

    def test_blocked_is_always_false_for_defects(self):
        client = _EchoA2Client()
        out = _collect_outcomes_a2(
            [_CLEAN_DIALOGUE], [_D1_RECORD], client=client, run_id="t",
            metric="blocked")
        assert out[0] is False
        assert out[1] is True
        assert client.arm == A2

    def test_clean_metric_only_returns_clean_samples(self):
        client = _EchoA2Client()
        out = _collect_outcomes_a2(
            [_CLEAN_DIALOGUE], [_D1_RECORD], client=client, run_id="t",
            metric="clean")
        assert out == [True]

    def test_detected_uses_deterministic_checks(self):
        client = _EchoA2Client()
        out = _collect_outcomes_a2(
            [_CLEAN_DIALOGUE], [_D1_RECORD], client=client, run_id="t",
            metric="detected")
        # check_consistency/结构规则看不出 D1，A2 不能凭空记检出
        assert out[0] is False

    def test_no_leak_uses_independent_judge(self):
        client = _EchoA2Client()
        out = _collect_outcomes_a2(
            [_CLEAN_DIALOGUE], [_D1_RECORD], client=client,
            judge=lambda _d: [{"rule": "C3_off_topic"}],
            run_id="t", metric="no_leak")
        assert out[0] is False, "判官判定仍跑题 → 缺陷泄露"

    def test_clean_intervention_detects_changed_content(self):
        def mutate(payload):
            payload["messages"][-1]["content"] = "基于文档更新，预计后日送达。"
            return payload

        r = {"ok": True, "dialogue": mutate(copy.deepcopy(_CLEAN_DIALOGUE))}
        assert a2_clean_intervened(r, _CLEAN_DIALOGUE) is True
        echo = {"ok": True, "dialogue": _CLEAN_DIALOGUE}
        assert a2_clean_intervened(echo, _CLEAN_DIALOGUE) is False
        assert a2_clean_intervened({"ok": False}, _CLEAN_DIALOGUE) is True


class TestJudgeFailureIsolation:
    def test_judge_error_is_not_detection_or_block(self):
        client = _ScriptClient([
            _call("check_consistency"),
            _call("check_coherence"),
            _call("fallback_keep"),
            {"text": "结束"},
        ])

        def boom(_dialogue):
            raise RuntimeError("judge down")

        result = run_arm(A4, [], [_D1_RECORD], client=client, judge=boom)
        row = result.per_type["D1_off_topic"]
        assert result.judge_error == 1
        assert row["total"] == 1
        assert row["detected"] == 0
        assert row["blocked"] == 0
        assert result.valid() is False  # 1/1 故障率超过 5%


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


class TestCustomInput:
    """--input/--defects：私有轨（无标签）与公开轨共用同一评测入口。"""

    @staticmethod
    def _write_base(tmp_path, n=3):
        dialogues = generate_dialogues(n=n, seed=7)
        p = tmp_path / "base.json"
        p.write_text(json.dumps(dialogues, ensure_ascii=False), encoding="utf-8")
        return p, dialogues

    @staticmethod
    def _write_defects(tmp_path, dialogues):
        records = generate(dialogues, per_type=1, seed=3)
        p = tmp_path / "defects.jsonl"
        p.write_text(to_jsonl(records), encoding="utf-8")
        return p, records

    def test_observational_mode_without_labels(self, tmp_path):
        base_path, _ = self._write_base(tmp_path)
        out = tmp_path / "out.json"
        code = main(["--arm", A3, "--input", str(base_path), "--out", str(out)])
        assert code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["run"]["data_mode"] == "observational"
        assert payload["run"]["n_defects"] == 0
        assert payload["clean_total"] == 3
        # 没有缺陷标签时不产出检出率结论
        assert payload["macro_recall_blocked"] == 0.0

    def test_limit_base_caps_clean_samples(self, tmp_path):
        base_path, _ = self._write_base(tmp_path, n=5)
        out = tmp_path / "out.json"
        code = main(["--arm", A3, "--input", str(base_path),
                     "--limit-base", "2", "--out", str(out)])
        assert code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["clean_total"] == 2

    def test_limit_base_rejected_when_defects_present(self, tmp_path):
        base_path, dialogues = self._write_base(tmp_path)
        defects_path, _ = self._write_defects(tmp_path, dialogues)
        code = main(["--arm", A3, "--input", str(base_path),
                     "--defects", str(defects_path), "--limit-base", "1"])
        assert code == 2

    def test_custom_defects_file_is_used(self, tmp_path):
        base_path, dialogues = self._write_base(tmp_path)
        defects_path, records = self._write_defects(tmp_path, dialogues)
        out = tmp_path / "out.json"
        code = main(["--arm", A3, "--input", str(base_path),
                     "--defects", str(defects_path), "--out", str(out)])
        assert code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["run"]["data_mode"] == "defects"
        assert payload["run"]["n_defects"] == len(records)

    def test_missing_input_returns_error(self, tmp_path):
        code = main(["--arm", A3, "--input", str(tmp_path / "absent.json")])
        assert code == 2


class TestReplayCLI:
    """replay 档必须完全离线：无密钥、无 --live，cassette 缺失时启动即失败。"""

    @staticmethod
    def _write_base(tmp_path, n=3):
        dialogues = generate_dialogues(n=n, seed=7)
        p = tmp_path / "base.json"
        p.write_text(json.dumps(dialogues, ensure_ascii=False), encoding="utf-8")
        return p

    def test_replay_requires_cassette_for_online_arm(self, tmp_path):
        base_path = self._write_base(tmp_path)
        code = main(["--arm", A3L, "--profile", "replay",
                     "--input", str(base_path)])
        assert code == 2

    def test_replay_rejects_live_flag(self, tmp_path):
        base_path = self._write_base(tmp_path)
        code = main(["--arm", A3L, "--profile", "replay", "--live",
                     "--cassette", str(tmp_path / "c.jsonl"),
                     "--input", str(base_path)])
        assert code == 2

    def test_replay_missing_cassette_returns_error(self, tmp_path):
        base_path = self._write_base(tmp_path)
        code = main(["--arm", A3L, "--profile", "replay",
                     "--cassette", str(tmp_path / "absent.jsonl"),
                     "--input", str(base_path)])
        assert code == 2

    def test_offline_arm_replay_runs_without_llm(self, tmp_path):
        base_path = self._write_base(tmp_path)
        out = tmp_path / "out.json"
        code = main(["--arm", A3, "--profile", "replay",
                     "--input", str(base_path), "--out", str(out)])
        assert code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["run"]["profile"] == "replay"
        assert payload["run"]["replayed_calls"] == 0


class TestCassetteIntegration:
    """A2 的录制与回放必须产出同一份指标，且回放不触发任何真实调用。"""

    def test_a2_record_then_replay_is_identical(self, dataset, tmp_path):
        from agent.runtime.cassette import (
            MODE_RECORD, MODE_REPLAY, CassetteClient,
        )

        base, defects = dataset
        path = tmp_path / "a2.jsonl"

        recorder = CassetteClient(path, mode=MODE_RECORD, inner=_EchoA2Client())
        live = run_arm(A2, base, defects, client=recorder)
        recorded = recorder.recorded
        recorder.close()
        assert recorded > 0

        replay = CassetteClient(path, mode=MODE_REPLAY)
        replayed = run_arm(A2, base, defects, client=replay)
        replay.close()

        assert replayed.per_type == live.per_type
        assert replayed.clean_flagged == live.clean_flagged
        assert replay.replayed == recorded
        # 回放全部命中缓存：不产生新的计费调用
        assert replayed.cost_cny == 0.0
