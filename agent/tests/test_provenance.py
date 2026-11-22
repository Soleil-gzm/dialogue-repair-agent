"""测试：模型溯源（provider 推断 / Ollama tags 解析 / manifest 结构）"""

from agent.benchmark.provenance import (
    infer_provider, llm_endpoint, model_provenance, ollama_model_info,
)
from agent.runtime.config import LLMSettings


TAGS = {
    "models": [
        {"name": "qwen3:8b", "model": "qwen3:8b", "digest": "sha256:abcdef0123456789",
         "size": 5_200_000_000, "modified_at": "2026-11-20T00:00:00Z",
         "details": {"family": "qwen3", "parameter_size": "8.2B",
                     "quantization_level": "Q4_K_M", "format": "gguf"}},
        {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "digest": "sha256:fedcba9876543210",
         "size": 4_700_000_000, "details": {}},
    ]
}


def _fetch(_url, _timeout):
    return TAGS


class TestProvider:
    def test_local_port_is_ollama(self):
        assert infer_provider("http://127.0.0.1:11434/v1") == "ollama"

    def test_cloud_url_keeps_default(self):
        assert infer_provider("https://api.deepseek.com") == "deepseek"

    def test_endpoint_strips_openai_suffix(self):
        assert llm_endpoint("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434"
        assert llm_endpoint("") is None


class TestOllamaModelInfo:
    def test_found_records_digest_and_quantization(self):
        info = ollama_model_info("http://127.0.0.1:11434/v1", "qwen3:8b",
                                 fetch_json=_fetch)
        assert info["found"] is True
        assert info["digest"] == "sha256:abcdef0123456789"
        assert info["details"]["quantization_level"] == "Q4_K_M"
        assert info["details"]["parameter_size"] == "8.2B"

    def test_tagless_model_matches_latest(self):
        tags = {"models": [{"name": "qwen2.5:latest", "model": "qwen2.5:latest",
                            "digest": "sha256:1"}]}
        info = ollama_model_info("http://127.0.0.1:11434/v1", "qwen2.5",
                                 fetch_json=lambda _u, _t: tags)
        assert info["found"] is True
        assert info["model"] == "qwen2.5:latest"

    def test_wrong_tag_is_not_fuzzy_matched(self):
        info = ollama_model_info("http://127.0.0.1:11434/v1", "qwen3",
                                 fetch_json=_fetch)
        assert info["found"] is False

    def test_missing_model_lists_available(self):
        info = ollama_model_info("http://127.0.0.1:11434/v1", "llama3:8b",
                                 fetch_json=_fetch)
        assert info["found"] is False
        assert "qwen3:8b" in info["available"]

    def test_network_failure_does_not_raise(self):
        def boom(_url, _timeout):
            raise OSError("connection refused")

        info = ollama_model_info("http://127.0.0.1:11434/v1", "qwen3:8b",
                                 fetch_json=boom)
        assert info["found"] is False
        assert "connection refused" in info["error"]


class TestModelProvenance:
    def test_ollama_settings_are_tagged_and_enriched(self):
        settings = LLMSettings(base_url="http://127.0.0.1:11434/v1",
                               model="qwen3:8b")
        info = model_provenance(settings, fetch_json=_fetch)
        assert info["provider"] == "ollama"
        assert info["model"] == "qwen3:8b"
        assert info["base_url"] == "http://127.0.0.1:11434/v1"
        assert info["ollama"]["found"] is True
        # 不落密钥本身，只留下环境变量名与是否已设置
        assert "api_key" not in info
        assert info["api_key_env"] == "DEEPSEEK_API_KEY"

    def test_cloud_settings_have_no_ollama_block(self):
        settings = LLMSettings(base_url="https://api.deepseek.com",
                               model="deepseek-chat")
        info = model_provenance(settings, fetch_json=_fetch)
        assert info["provider"] == "deepseek"
        assert "ollama" not in info
