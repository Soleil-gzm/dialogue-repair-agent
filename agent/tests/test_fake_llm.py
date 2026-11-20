"""测试：FakeLLMClient

假客户端的存在意义：让工具与循环逻辑可以脱离网络单测，
并让 Benchmark 在无密钥环境下可复现。
"""

from agent.runtime.llm import FakeLLMClient, LLMResponse, StageRoutedClient


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


class TestStageRouting:
    def test_routes_judge_stage(self):
        gen, judge = FakeLLMClient(default="gen"), FakeLLMClient(default="judge")
        c = StageRoutedClient(gen, judge)
        msgs = [{"role": "user", "content": "x"}]

        assert c.chat(msgs, stage="judge").text == "judge"
        assert c.chat(msgs, stage="generate").text == "gen"
        assert c.chat(msgs).text == "gen"
        assert gen.call_count == 2 and judge.call_count == 1

    def test_judge_defaults_to_generate(self):
        gen = FakeLLMClient(default="gen")
        c = StageRoutedClient(gen)
        assert c.chat([{"role": "user", "content": "x"}], stage="judge").text == "gen"

    def test_set_arm_reaches_both_clients(self):
        class Recorder:
            def __init__(self):
                self.arm = "adhoc"

            def set_arm(self, arm):
                self.arm = arm

            def chat(self, messages, **kwargs):
                return LLMResponse(text="{}")

        gen, judge = Recorder(), Recorder()
        StageRoutedClient(gen, judge).set_arm("A4")
        assert (gen.arm, judge.arm) == ("A4", "A4")

    def test_build_stage_clients_uses_judge_model(self):
        from agent.runtime.deepseek import build_stage_clients

        env = {"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_MODEL": "gen",
               "JUDGE_MODEL": "judge-model"}
        c = build_stage_clients(environ=env)
        assert c.generate_client.settings.model == "gen"
        assert c.judge_client.settings.model == "judge-model"
        assert c.judge_client.settings.temperature == 0.0
