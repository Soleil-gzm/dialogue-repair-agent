"""测试：DeepSeekClient（全离线，用 MockTransport，不需要网络与 key）"""

import json

import httpx
import pytest

from agent.runtime.config import LLMSettings
from agent.runtime.deepseek import (
    DeepSeekClient, LLMHTTPError, LLMParseError,
)
from agent.runtime.ledger import Pricing, TokenLedger

OK_BODY = {
    "model": "deepseek-chat",
    "choices": [{"message": {"content": "{\"ok\": true}"}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
}


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
        c = _client(_ok)
        r = c.chat([{"role": "user", "content": "hi"}])
        assert r.text == '{"ok": true}'
        assert r.input_tokens == 120 and r.output_tokens == 30
        assert r.cached is False

    def test_sends_bearer_token(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("Authorization")
            seen["path"] = request.url.path
            return httpx.Response(200, json=OK_BODY)

        _client(handler).chat([{"role": "user", "content": "x"}])
        assert seen["auth"] == "Bearer sk-test"
        assert seen["path"].endswith("/chat/completions")

    def test_json_mode_adds_response_format(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=OK_BODY)

        _client(handler).chat([{"role": "user", "content": "x"}])
        assert seen["body"]["response_format"] == {"type": "json_object"}

    def test_tools_passed_through(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=OK_BODY)

        specs = [{"type": "function", "function": {"name": "t"}}]
        _client(handler).chat([{"role": "user", "content": "x"}], tools=specs)
        assert seen["body"]["tools"] == specs


class TestCaching:
    def test_second_identical_call_is_cached(self):
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(200, json=OK_BODY)

        c = _client(handler)
        first = c.chat([{"role": "user", "content": "same"}])
        second = c.chat([{"role": "user", "content": "same"}])
        assert calls["n"] == 1
        assert first.cached is False and second.cached is True

    def test_cached_call_costs_zero(self):
        ledger = TokenLedger()
        c = _client(_ok, ledger=ledger,
                    pricing=Pricing(input_per_1m=1000, output_per_1m=2000, version="v1"))
        c.chat([{"role": "user", "content": "same"}])
        c.chat([{"role": "user", "content": "same"}])
        assert ledger.entries[0].cost_cny > 0
        assert ledger.entries[1].cached is True
        assert ledger.entries[1].cost_cny == 0.0

    def test_different_prompt_not_cached(self):
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(200, json=OK_BODY)

        c = _client(handler)
        c.chat([{"role": "user", "content": "a"}])
        c.chat([{"role": "user", "content": "b"}])
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
        r = _client(handler, ledger=ledger).chat([{"role": "user", "content": "x"}])
        assert r.text == '{"ok": true}'
        assert ledger.entries[0].retries == 1

    def test_retries_on_transport_error(self):
        seq = {"n": 0}

        def handler(request):
            seq["n"] += 1
            if seq["n"] < 2:
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(200, json=OK_BODY)

        r = _client(handler).chat([{"role": "user", "content": "x"}])
        assert r.text == '{"ok": true}'

    def test_does_not_retry_on_400(self):
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(400, text="bad request")

        with pytest.raises(LLMHTTPError):
            _client(handler).chat([{"role": "user", "content": "x"}])
        assert calls["n"] == 1, "参数错误重试没有意义"

    def test_gives_up_after_max_attempts(self):
        calls = {"n": 0}

        def handler(_request):
            calls["n"] += 1
            return httpx.Response(503, text="unavailable")

        with pytest.raises(LLMHTTPError):
            _client(handler, max_attempts=3).chat([{"role": "user", "content": "x"}])
        assert calls["n"] == 3

    def test_failed_call_recorded_in_ledger(self):
        ledger = TokenLedger()
        with pytest.raises(LLMHTTPError):
            _client(lambda _r: httpx.Response(400, text="bad"), ledger=ledger).chat(
                [{"role": "user", "content": "x"}])
        e = ledger.entries[0]
        assert e.ok is False
        assert e.error_type == "LLMHTTPError"
        assert e.http_status == 400


class TestParseErrors:
    def test_missing_choices_raises(self):
        c = _client(lambda _r: httpx.Response(200, json={"model": "m"}))
        with pytest.raises(LLMParseError):
            c.chat([{"role": "user", "content": "x"}])

    def test_invalid_json_raises(self):
        c = _client(lambda _r: httpx.Response(200, text="<html>"))
        with pytest.raises(LLMParseError):
            c.chat([{"role": "user", "content": "x"}])

    def test_missing_usage_defaults_zero(self):
        body = {"choices": [{"message": {"content": "hi"}}]}
        c = _client(lambda _r: httpx.Response(200, json=body))
        r = c.chat([{"role": "user", "content": "x"}])
        assert r.input_tokens == 0 and r.output_tokens == 0
