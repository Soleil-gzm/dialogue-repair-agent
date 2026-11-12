"""测试：DeepSeekClient（全离线，用 MockTransport，不需要网络与 key）"""

import json

import httpx
import pytest

from agent.runtime.config import LLMSettings
from agent.runtime.deepseek import (
    DeepSeekClient, LLMConfigError, LLMHTTPError, LLMParseError,
)
from agent.runtime.ledger import Pricing, TokenLedger
from agent.runtime.pricing import PEAK

OK_BODY = {
    "model": "deepseek-chat",
    "choices": [{"message": {"content": '{"ok": true}'}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
}

# json_mode 要求 prompt 含 "json" 字样（DeepSeek 接口约束）
PROMPT = [{"role": "user", "content": "请输出 json"}]


def _settings() -> LLMSettings:
    return LLMSettings(model="deepseek-chat", base_url="https://api.test")


def _client(handler, **kw) -> DeepSeekClient:
    return DeepSeekClient(
        settings=_settings(),
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
        backoff_base=0.0,   # 测试不等待
        **kw,
    )


def _ok(_request):
    return httpx.Response(200, json=OK_BODY)


class TestSuccessPath:
    def test_returns_text_and_usage(self):
        r = _client(_ok).chat(PROMPT)
        assert r.text == '{"ok": true}'
        assert r.input_tokens == 120 and r.output_tokens == 30
        assert r.cached is False

    def test_sends_bearer_token(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("Authorization")
            seen["path"] = request.url.path
            return httpx.Response(200, json=OK_BODY)

        _client(handler).chat(PROMPT)
        assert seen["auth"] == "Bearer sk-test"
        assert seen["path"].endswith("/chat/completions")

    def test_json_mode_adds_response_format(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=OK_BODY)

        _client(handler).chat(PROMPT)
        assert seen["body"]["response_format"] == {"type": "json_object"}

    def test_tools_passed_through(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=OK_BODY)

        specs = [{"type": "function", "function": {"name": "t"}}]
        _client(handler).chat(PROMPT, tools=specs)
        assert seen["body"]["tools"] == specs


class TestUsageAccounting:
    def test_parses_deepseek_cache_hit_tokens(self):
        body = {
            "model": "deepseek-flash",
            "choices": [{"message": {"content": "{}"}}],
            "usage": {
                "prompt_tokens": 1000, "completion_tokens": 100,
                "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200,
            },
        }
        r = _client(lambda _r: httpx.Response(200, json=body)).chat(PROMPT)
        assert r.cached_tokens == 800

    def test_parses_openai_style_cached_tokens(self):
        body = {
            "model": "m",
            "choices": [{"message": {"content": "{}"}}],
            "usage": {
                "prompt_tokens": 1000, "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 600},
            },
        }
        r = _client(lambda _r: httpx.Response(200, json=body)).chat(PROMPT)
        assert r.cached_tokens == 600

    def test_ledger_cost_applies_cache_hit_price(self):
        body = {
            "model": "deepseek-flash",
            "choices": [{"message": {"content": "{}"}}],
            "usage": {
                "prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                "prompt_cache_hit_tokens": 1_000_000,
            },
        }
        ledger = TokenLedger()
        _client(lambda _r: httpx.Response(200, json=body),
                ledger=ledger, pricing=PEAK).chat(PROMPT)
        e = ledger.entries[0]
        assert e.cached_tokens == 1_000_000
        assert e.cost_cny == pytest.approx(0.04 + 8.0)
        assert e.price_table_version == PEAK.version


class TestJsonModeGuard:
    def test_rejects_prompt_without_json_keyword(self):
        """DeepSeek 的硬约束在本地拦截，不必等服务端 400。"""
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(200, json=OK_BODY)

        with pytest.raises(LLMConfigError):
            _client(handler).chat([{"role": "user", "content": "请输出结果"}])
        assert calls["n"] == 0, "不该发出请求"

    def test_allows_when_response_format_absent_and_json_mode_off(self):
        s = LLMSettings(model="m", base_url="https://api.test", json_mode=False)
        c = DeepSeekClient(settings=s, api_key="k",
                           transport=httpx.MockTransport(_ok))
        assert c.chat([{"role": "user", "content": "任意"}]).text == '{"ok": true}'


class TestCaching:
    def test_second_identical_call_is_cached(self):
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(200, json=OK_BODY)

        c = _client(handler)
        first = c.chat(PROMPT)
        second = c.chat(PROMPT)
        assert calls["n"] == 1
        assert first.cached is False and second.cached is True

    def test_cached_call_costs_zero(self):
        ledger = TokenLedger()
        c = _client(_ok, ledger=ledger,
                    pricing=Pricing(input_per_1m=1000, output_per_1m=2000, version="v1"))
        c.chat(PROMPT)
        c.chat(PROMPT)
        assert ledger.entries[0].cost_cny > 0
        assert ledger.entries[1].cached is True
        assert ledger.entries[1].cost_cny == 0.0

    def test_different_prompt_not_cached(self):
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(200, json=OK_BODY)

        c = _client(handler)
        c.chat([{"role": "user", "content": "a json"}])
        c.chat([{"role": "user", "content": "b json"}])
        assert calls["n"] == 2


class TestRetry:
    def test_retries_on_500_then_succeeds(self):
        seq = {"n": 0}

        def handler(_request):
            seq["n"] += 1
            if seq["n"] < 2:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json=OK_BODY)

        ledger = TokenLedger()
        r = _client(handler, ledger=ledger).chat(PROMPT)
        assert r.text == '{"ok": true}'
        assert ledger.entries[0].retries == 1

    def test_retries_on_transport_error(self):
        seq = {"n": 0}

        def handler(request):
            seq["n"] += 1
            if seq["n"] < 2:
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(200, json=OK_BODY)

        assert _client(handler).chat(PROMPT).text == '{"ok": true}'

    def test_does_not_retry_on_400(self):
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(400, text="bad request")

        with pytest.raises(LLMHTTPError):
            _client(handler).chat(PROMPT)
        assert calls["n"] == 1, "参数错误重试没有意义"

    def test_gives_up_after_max_attempts(self):
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(503, text="unavailable")

        with pytest.raises(LLMHTTPError):
            _client(handler, max_attempts=3).chat(PROMPT)
        assert calls["n"] == 3

    def test_failed_call_recorded_in_ledger(self):
        ledger = TokenLedger()
        with pytest.raises(LLMHTTPError):
            _client(lambda _r: httpx.Response(400, text="bad"), ledger=ledger).chat(PROMPT)
        e = ledger.entries[0]
        assert e.ok is False
        assert e.error_type == "LLMHTTPError"
        assert e.http_status == 400


class TestToolCalls:
    TOOL_BODY = {
        "model": "deepseek-flash",
        "choices": [{"message": {
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "check_consistency",
                             "arguments": '{"before": {"messages": []}, "after": {}}'},
            }],
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }

    def test_parses_tool_calls(self):
        r = _client(lambda _r: httpx.Response(200, json=self.TOOL_BODY)).chat(
            [{"role": "user", "content": "任意"}], tools=[{"type": "function"}])
        assert r.tool_calls[0]["name"] == "check_consistency"
        assert r.tool_calls[0]["arguments"] == {"before": {"messages": []}, "after": {}}
        assert r.text == ""

    def test_json_mode_not_forced_when_tools_present(self):
        """工具调用与 json_object 互斥，带 tools 时不能加 response_format。"""
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=self.TOOL_BODY)

        _client(handler).chat([{"role": "user", "content": "无需 json 字样"}],
                              tools=[{"type": "function"}])
        assert "response_format" not in seen["body"]
        assert seen["body"]["tools"] == [{"type": "function"}]

    def test_cached_response_keeps_tool_calls(self):
        c = _client(lambda _r: httpx.Response(200, json=self.TOOL_BODY))
        args = ([{"role": "user", "content": "任意"}], {"tools": [{"type": "function"}]})
        c.chat(args[0], tools=args[1]["tools"])
        second = c.chat(args[0], tools=args[1]["tools"])
        assert second.cached is True
        assert second.tool_calls[0]["name"] == "check_consistency"

    def test_malformed_arguments_becomes_empty_dict(self):
        body = {
            "choices": [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "t", "arguments": "not-json"}},
            ]}}],
        }
        r = _client(lambda _r: httpx.Response(200, json=body)).chat(
            [{"role": "user", "content": "x"}], tools=[{"type": "function"}])
        assert r.tool_calls[0]["arguments"] == {}

    def test_tool_call_without_name_skipped(self):
        body = {"choices": [{"message": {"content": "", "tool_calls": [
            {"function": {"arguments": "{}"}},
        ]}}]}
        r = _client(lambda _r: httpx.Response(200, json=body)).chat(
            [{"role": "user", "content": "x"}], tools=[{"type": "function"}])
        assert r.tool_calls == []


class TestParseErrors:
    def test_missing_choices_raises(self):
        c = _client(lambda _r: httpx.Response(200, json={"model": "m"}))
        with pytest.raises(LLMParseError):
            c.chat(PROMPT)

    def test_invalid_json_raises(self):
        c = _client(lambda _r: httpx.Response(200, text="<html>"))
        with pytest.raises(LLMParseError):
            c.chat(PROMPT)

    def test_missing_usage_defaults_zero(self):
        body = {"choices": [{"message": {"content": "hi"}}]}
        r = _client(lambda _r: httpx.Response(200, json=body)).chat(PROMPT)
        assert r.input_tokens == 0 and r.output_tokens == 0
