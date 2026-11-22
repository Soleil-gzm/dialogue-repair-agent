"""测试：LLM 响应录制与回放

为什么需要：replay 用于离线复核在线结论，指纹必须只由影响模型输出的参数决定，
否则 run_id 等审计字段变化会让回放全部 miss。
"""

import json

import pytest

from agent.runtime.cassette import (
    MODE_RECORD, MODE_REPLAY, CassetteClient, CassetteMiss, request_key,
)
from agent.runtime.llm import LLMResponse


class _Inner:
    def __init__(self, text="答复", model="fake-model", tool_calls=None):
        self.text = text
        self.model = model
        self.tool_calls = tool_calls or []
        self.calls = []
        self.arm = "adhoc"
        self.closed = False

    def set_arm(self, arm):
        self.arm = arm

    def close(self):
        self.closed = True

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(
            text=self.text, model=self.model,
            input_tokens=11, output_tokens=7,
            tool_calls=list(self.tool_calls),
        )


_MSGS = [{"role": "user", "content": "你好"}]


class TestRequestKey:
    def test_ignores_audit_fields(self):
        """stage/run_id/dialogue_key 不影响指纹，否则回放永远匹配不上。"""
        k1 = request_key(_MSGS, temperature=0.3, max_tokens=64)
        # 参数对象相同即同指纹；审计字段不在 request_key 入参里
        k2 = request_key(_MSGS, temperature=0.3, max_tokens=64)
        assert k1 == k2

    def test_changes_with_semantics(self):
        base = request_key(_MSGS, temperature=0.0, max_tokens=64)
        assert base != request_key(_MSGS, temperature=0.3, max_tokens=64)
        assert base != request_key(_MSGS, temperature=0.0, max_tokens=128)
        assert base != request_key(
            [{"role": "user", "content": "再见"}], temperature=0.0, max_tokens=64)

    def test_tools_change_key(self):
        specs = [{"type": "function", "function": {"name": "check"}}]
        base = request_key(_MSGS, temperature=0.0, max_tokens=64)
        assert base != request_key(
            _MSGS, temperature=0.0, max_tokens=64, tools=specs)

    def test_dict_order_is_stable(self):
        a = request_key([{"role": "user", "content": "x"}],
                        temperature=0.0, max_tokens=8)
        b = request_key([{"content": "x", "role": "user"}],
                        temperature=0.0, max_tokens=8)
        assert a == b


class TestRecord:
    def test_writes_jsonl_and_returns_live_response(self, tmp_path):
        path = tmp_path / "cassette.jsonl"
        inner = _Inner()
        client = CassetteClient(path, mode=MODE_RECORD, inner=inner)
        resp = client.chat(_MSGS, temperature=0.3, max_tokens=64)
        client.close()

        assert resp.text == "答复" and resp.cached is False
        assert inner.calls and inner.closed is True
        records = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 1
        assert records[0]["request"]["messages"] == _MSGS
        assert records[0]["response"]["input_tokens"] == 11
        assert records[0]["meta"]["stage"] == "check"

    def test_requires_inner(self, tmp_path):
        with pytest.raises(ValueError):
            CassetteClient(tmp_path / "c.jsonl", mode=MODE_RECORD)

    def test_rejects_unknown_mode(self, tmp_path):
        with pytest.raises(ValueError):
            CassetteClient(tmp_path / "c.jsonl", mode="nope")

    def test_set_arm_forwards(self, tmp_path):
        inner = _Inner()
        client = CassetteClient(tmp_path / "c.jsonl", mode=MODE_RECORD, inner=inner)
        client.set_arm("A4")
        assert client.arm == "A4" and inner.arm == "A4"
        client.close()


class TestReplay:
    def _record(self, tmp_path, **chat_kwargs):
        path = tmp_path / "cassette.jsonl"
        client = CassetteClient(path, mode=MODE_RECORD, inner=_Inner())
        client.chat(_MSGS, **chat_kwargs)
        client.close()
        return path

    def test_returns_recorded_response_without_network(self, tmp_path):
        path = self._record(tmp_path, temperature=0.3, max_tokens=64)
        # 回放不需要 inner：没有网络访问路径
        client = CassetteClient(path, mode=MODE_REPLAY)
        resp = client.chat(_MSGS, temperature=0.3, max_tokens=64)
        client.close()

        assert resp.text == "答复"
        assert resp.input_tokens == 11 and resp.output_tokens == 7
        assert resp.cached is True, "回放必须标记 cached，账本成本才能为 0"
        assert resp.latency_ms == 0
        assert client.replayed == 1

    def test_run_id_and_stage_do_not_break_replay(self, tmp_path):
        path = self._record(tmp_path, temperature=0.0, max_tokens=16)
        client = CassetteClient(path, mode=MODE_REPLAY)
        resp = client.chat(_MSGS, temperature=0.0, max_tokens=16,
                           stage="judge", run_id="another-run",
                           dialogue_key="other")
        client.close()
        assert resp.text == "答复"

    def test_miss_raises_clear_error(self, tmp_path):
        path = self._record(tmp_path, temperature=0.0, max_tokens=16)
        client = CassetteClient(path, mode=MODE_REPLAY)
        try:
            with pytest.raises(CassetteMiss) as err:
                client.chat([{"role": "user", "content": "不同请求"}],
                            temperature=0.0, max_tokens=16)
            assert "cassette" in str(err.value)
        finally:
            client.close()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CassetteMiss):
            CassetteClient(tmp_path / "absent.jsonl", mode=MODE_REPLAY)

    def test_tool_calls_round_trip(self, tmp_path):
        path = tmp_path / "cassette.jsonl"
        calls = [{"id": "c1", "name": "check_consistency", "arguments": {"x": 1}}]
        inner = _Inner(text="", tool_calls=calls)
        recorder = CassetteClient(path, mode=MODE_RECORD, inner=inner)
        recorder.chat(_MSGS, temperature=0.0, max_tokens=8)
        recorder.close()

        client = CassetteClient(path, mode=MODE_REPLAY)
        resp = client.chat(_MSGS, temperature=0.0, max_tokens=8)
        client.close()
        assert resp.tool_calls == calls

    def test_entries_counts_loaded_records(self, tmp_path):
        path = self._record(tmp_path, temperature=0.0, max_tokens=8)
        client = CassetteClient(path, mode=MODE_REPLAY)
        assert client.entries == 1
        client.close()
