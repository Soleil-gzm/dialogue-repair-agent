"""测试：LLMPolicy（function calling 决策，全离线）

用桩客户端验证"决策契约"，不验证模型好坏；真实模型表现留给 benchmark 测量。
"""

import copy
import random

import pytest

from agent.benchmark.defects import inject
from agent.runtime.budget import Budget
from agent.runtime.llm import LLMResponse
from agent.runtime.loop import ACCEPTED, FALLBACK, AgentRuntime
from agent.runtime.policy import (
    ACTION_CALL_TOOL, ACTION_STOP, LLMPolicy, LLMPolicyError,
)
from agent.runtime.state import AgentState
from agent.tools import register_all


class _StubClient:
    """按脚本返回响应，并记录每次调用参数。"""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        item = self._responses.pop(0) if self._responses else {}
        return LLMResponse(
            text=item.get("text", ""),
            tool_calls=item.get("tool_calls", []),
        )


def _call(name):
    return {"tool_calls": [{"id": "c1", "name": name, "arguments": {}}]}


def _state(dialogue, issues=None, history=None):
    return AgentState(
        dialogue=dialogue,
        original=dialogue,
        budget=Budget(),
        issues=list(issues or []),
        history=list(history or []),
    )


@pytest.fixture(autouse=True)
def tools_registered():
    register_all()


class TestDecide:
    def test_model_picks_tool(self, mini_dialogues):
        d = mini_dialogues[0]
        client = _StubClient(_call("check_consistency"))
        dec = LLMPolicy(client).decide(_state(d))
        assert dec.action == ACTION_CALL_TOOL
        assert dec.tool == "check_consistency"
        assert dec.args == {"before": d, "after": d}
        assert dec.reason

    def test_model_can_stop_with_reason(self, mini_dialogues):
        client = _StubClient({"text": "两个校验都通过，接受当前版本"})
        dec = LLMPolicy(client).decide(_state(mini_dialogues[0]))
        assert dec.action == ACTION_STOP
        assert "接受" in dec.reason

    def test_args_come_from_state_not_model(self, mini_dialogues):
        """模型给的参数不可信：它只负责选工具，参数由运行时注入。"""
        d = mini_dialogues[0]
        client = _StubClient({"tool_calls": [{"name": "check_consistency",
                                              "arguments": {"before": {"messages": []}}}]})
        dec = LLMPolicy(client).decide(_state(d))
        assert dec.args["before"] == d, "不能采用模型自带的 before"
        assert dec.args["after"] == d

    def test_coherence_args(self, mini_dialogues):
        d = mini_dialogues[0]
        client = _StubClient(_call("check_coherence"))
        dec = LLMPolicy(client).decide(_state(d))
        assert dec.args == {"dialogue": d}

    def test_fallback_args(self, mini_dialogues):
        d = mini_dialogues[0]
        client = _StubClient(_call("fallback_keep"))
        dec = LLMPolicy(client).decide(_state(d))
        assert dec.args == {"original": d}

    def test_reason_truncated(self, mini_dialogues):
        client = _StubClient({"text": "很长的理由" * 100})
        dec = LLMPolicy(client).decide(_state(mini_dialogues[0]))
        assert len(dec.reason) <= 200


class TestContract:
    def test_unknown_tool_raises(self, mini_dialogues):
        client = _StubClient(_call("rm_rf"))
        with pytest.raises(LLMPolicyError, match="未注册"):
            LLMPolicy(client).decide(_state(mini_dialogues[0]))

    def test_repeated_tool_raises(self, mini_dialogues):
        d = mini_dialogues[0]
        client = _StubClient(_call("check_consistency"))
        with pytest.raises(LLMPolicyError, match="重复调用"):
            LLMPolicy(client).decide(
                _state(d, history=[{"tool": "check_consistency", "ok": True}]))

    def test_empty_response_raises(self, mini_dialogues):
        """既不调用工具也不给理由，不能被当成"通过"。"""
        with pytest.raises(LLMPolicyError, match="结束理由"):
            LLMPolicy(_StubClient()).decide(_state(mini_dialogues[0]))

    def test_restricted_tool_set(self, mini_dialogues):
        client = _StubClient(_call("check_coherence"))
        with pytest.raises(LLMPolicyError):
            LLMPolicy(client, tools=["check_consistency"]).decide(
                _state(mini_dialogues[0]))


class TestRequestShape:
    def test_exposes_tools_with_empty_parameters(self, mini_dialogues):
        client = _StubClient({"text": "ok"})
        LLMPolicy(client).decide(_state(mini_dialogues[0]))
        specs = client.calls[0]["tools"]
        names = {s["function"]["name"] for s in specs}
        assert {"check_consistency", "check_coherence", "fallback_keep"} <= names
        for s in specs:
            assert s["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_zero_temperature_and_stage(self, mini_dialogues):
        client = _StubClient({"text": "ok"})
        LLMPolicy(client, run_id="run-7").decide(_state(mini_dialogues[0]))
        call = client.calls[0]
        assert call["temperature"] == 0.0
        assert call["stage"] == "decide"
        assert call["run_id"] == "run-7"
        assert len(call["dialogue_key"]) == 16

    def test_observation_has_no_dialogue_content(self, mini_dialogues):
        """观测只给决策所需信息：不允许把整段对话塞进 prompt。"""
        d = dict(mini_dialogues[0])
        d["messages"] = [dict(m, content="敏感内容" * 50) for m in d["messages"]]
        client = _StubClient({"text": "ok"})
        LLMPolicy(client).decide(_state(d))
        user = client.calls[0]["messages"][1]["content"]
        assert "敏感内容" not in user
        assert "消息数" in user

    def test_observation_lists_history_and_issues(self, mini_dialogues):
        d = mini_dialogues[0]
        state = _state(d,
                       issues=[{"rule": "R3_entity_fidelity", "turn": 3, "detail": "实体丢失: X"}],
                       history=[{"tool": "check_consistency", "ok": False}])
        client = _StubClient({"text": "ok"})
        LLMPolicy(client).decide(state)
        user = client.calls[0]["messages"][1]["content"]
        assert "check_consistency(有违规)" in user
        assert "R3_entity_fidelity" in user


class TestIntegrationWithLoop:
    def test_clean_dialogue_accepted(self, mini_dialogues):
        client = _StubClient(_call("check_consistency"), _call("check_coherence"),
                             {"text": "校验通过"})
        r = AgentRuntime(LLMPolicy(client)).run(mini_dialogues[0])
        assert r.status == ACCEPTED
        assert r.tool_calls == 2
        assert client.calls[0]["stage"] == "decide"

    def test_defective_dialogue_falls_back(self, mini_dialogues):
        clean = mini_dialogues[0]
        bad, _ = inject(clean, "D3_fact_conflict", random.Random(1))
        client = _StubClient(_call("check_consistency"), _call("fallback_keep"))
        r = AgentRuntime(LLMPolicy(client)).run(bad, reference=clean)
        assert r.status == FALLBACK
        assert r.dialogue == clean
        assert r.issues

    def test_policy_error_marks_failed(self, mini_dialogues):
        client = _StubClient(_call("rm_rf"))
        r = AgentRuntime(LLMPolicy(client)).run(mini_dialogues[0])
        assert r.status == "failed"
        assert "策略异常" in r.reason

    def test_does_not_mutate_state_dialogue(self, mini_dialogues):
        before = copy.deepcopy(mini_dialogues[0])
        client = _StubClient({"text": "ok"})
        LLMPolicy(client).decide(_state(mini_dialogues[0]))
        assert mini_dialogues[0] == before
