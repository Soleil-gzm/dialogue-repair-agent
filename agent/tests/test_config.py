"""测试：配置与密钥读取

重点不是"能读到值"，而是"读不到时明确失败、且不泄漏"。
"""

import pytest

from agent.runtime.config import (
    LLMSettings, MissingCredentials, load_dotenv, parse_dotenv,
)


class TestParseDotenv:
    def test_basic_pairs(self):
        assert parse_dotenv("A=1\nB=two") == {"A": "1", "B": "two"}

    def test_ignores_comments_and_blanks(self):
        text = "# 注释\n\nA=1\n   \n# B=2\n"
        assert parse_dotenv(text) == {"A": "1"}

    def test_strips_quotes(self):
        assert parse_dotenv('A="x y"\nB=\'z\'') == {"A": "x y", "B": "z"}

    def test_keeps_empty_value(self):
        assert parse_dotenv("A=") == {"A": ""}

    def test_ignores_malformed_line(self):
        assert parse_dotenv("不是键值对") == {}


class TestLoadDotenv:
    def test_missing_file_is_silent(self, tmp_path):
        assert load_dotenv(tmp_path / "nope.env") == {}

    def test_loads_into_environ(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("AGENT_TEST_KEY=abc\n", encoding="utf-8")
        monkeypatch.delenv("AGENT_TEST_KEY", raising=False)
        load_dotenv(p)
        import os
        assert os.environ["AGENT_TEST_KEY"] == "abc"

    def test_does_not_override_existing_by_default(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("AGENT_TEST_KEY=from_file\n", encoding="utf-8")
        monkeypatch.setenv("AGENT_TEST_KEY", "from_env")
        load_dotenv(p)
        import os
        assert os.environ["AGENT_TEST_KEY"] == "from_env"


class TestLLMSettings:
    def test_from_env_overrides_model_and_url(self):
        s = LLMSettings.from_env({
            "DEEPSEEK_MODEL": "m2", "DEEPSEEK_BASE_URL": "https://x",
        })
        assert s.model == "m2" and s.base_url == "https://x"

    def test_api_key_missing_raises(self):
        with pytest.raises(MissingCredentials):
            LLMSettings.from_env({}).api_key({})

    def test_api_key_blank_raises(self):
        with pytest.raises(MissingCredentials):
            LLMSettings.from_env({"DEEPSEEK_API_KEY": "   "}).api_key(
                {"DEEPSEEK_API_KEY": "   "})

    def test_api_key_returns_value(self):
        env = {"DEEPSEEK_API_KEY": "sk-test"}
        assert LLMSettings.from_env(env).api_key(env) == "sk-test"

    def test_redacted_never_contains_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
        view = LLMSettings.from_env().redacted()
        assert view["has_key"] is True
        assert "sk-secret" not in str(view)
        assert "api_key" not in view

    def test_has_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert LLMSettings.from_env().has_key() is False
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        assert LLMSettings.from_env().has_key() is True
