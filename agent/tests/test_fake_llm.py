"""测试：FakeLLMClient

假客户端的存在意义：让工具与循环逻辑可以脱离网络单测，
并让 Benchmark 在无密钥环境下可复现。
"""

from agent.runtime.llm import FakeLLMClient, LLMResponse


class TestFakeClient:
    def test_returns_default(self):
        c = FakeLLMClient(default='{"ok": true}')
        r = c.chat([{"role": "user", "content": "hi"}])
        assert r.text == '{"ok": true}'
        assert isinstance(r, LLMResponse)

    def test_deterministic_for_same_input(self):
        c = FakeLLMClient(default="x")
        msgs = [{"role": "user", "content": "同一段输入"}]
        assert c.chat(msgs).text == c.chat(msgs).text

    def test_match_by_message_hash(self):
        msgs = [{"role": "user", "content": "特定输入"}]
        c = FakeLLMClient(default="fallback")
        key = FakeLLMClient._hash(msgs)
        c2 = FakeLLMClient(responses={key: "命中"}, default="fallback")
        assert c2.chat(msgs).text == "命中"

    def test_match_by_idempotency_key_has_priority(self):
        msgs = [{"role": "user", "content": "任意"}]
        c = FakeLLMClient(responses={"k1": "按key命中"}, default="fallback")
        assert c.chat(msgs, idempotency_key="k1").text == "按key命中"

    def test_records_calls(self):
        c = FakeLLMClient()
        c.chat([{"role": "user", "content": "a"}], temperature=0.3, max_tokens=64)
        assert c.call_count == 1
        assert c.calls[0]["temperature"] == 0.3
        assert c.calls[0]["max_tokens"] == 64

    def test_token_estimation_non_zero(self):
        c = FakeLLMClient(default="这是一段返回内容")
        r = c.chat([{"role": "user", "content": "这是一段输入内容"}])
        assert r.input_tokens > 0
        assert r.output_tokens > 0
        assert r.total_tokens == r.input_tokens + r.output_tokens

    def test_cached_defaults_false(self):
        r = FakeLLMClient().chat([{"role": "user", "content": "x"}])
        assert r.cached is False

    def test_passes_tools_through(self):
        c = FakeLLMClient()
        specs = [{"type": "function", "function": {"name": "t"}}]
        c.chat([{"role": "user", "content": "x"}], tools=specs)
        assert c.calls[0]["tools"] == specs
