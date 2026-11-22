"""模型溯源：记录结果由哪个 provider / 模型 / 权重跑出来

公司电脑与本地电脑的 Ollama 模型可能不同，单写一个模型名无法区分身份。
因此 manifest 除模型名外，还要落盘接入点与模型文件指纹（digest）。

约束：
- 只做记录，查询失败不阻断评测（网络/服务不可用属正常情况）
- 不写密钥：复用 `LLMSettings.redacted()`
"""

import json
import urllib.request
from typing import Any, Callable, Dict, Optional

# Ollama 原生 API 默认端口；OpenAI 兼容层通常挂在 `<root>/v1`
OLLAMA_PORT = "11434"

# 允许测试注入，避免真发网络请求
FetchJSON = Callable[[str, float], Dict[str, Any]]


def infer_provider(base_url: str, default: str = "deepseek") -> str:
    """从接入点推断 provider；Ollama 走本地端口或名字里带 ollama。"""
    low = (base_url or "").lower()
    if OLLAMA_PORT in low or "ollama" in low:
        return "ollama"
    return default


def llm_endpoint(base_url: str) -> Optional[str]:
    """OpenAI 兼容地址 → Ollama 原生 API 根（去掉末尾 /v1）。"""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return None
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _fetch_json(url: str, timeout_s: float) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize(name: str) -> str:
    """Ollama 省略 tag 时等价于 `:latest`，比较前补齐。"""
    n = (name or "").strip()
    return n if ":" in n else f"{n}:latest"


def ollama_model_info(
    base_url: str,
    model: str,
    *,
    fetch_json: Optional[FetchJSON] = None,
    timeout_s: float = 2.0,
) -> Dict[str, Any]:
    """查询 Ollama `/api/tags`，返回目标模型的 digest 与量化信息。

    失败时返回 `{"found": False, "error": ...}`，调用方不得因此中断评测。
    """
    root = llm_endpoint(base_url)
    if not root:
        return {"found": False, "error": "base_url 为空"}
    api = f"{root}/api/tags"
    try:
        data = (fetch_json or _fetch_json)(api, timeout_s)
    except Exception as exc:  # noqa: BLE001 - 溯源失败不能影响主流程
        return {"found": False, "api": api, "error": f"{type(exc).__name__}: {exc}"}

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return {"found": False, "api": api, "error": "响应缺少 models 列表"}

    target = _normalize(model)
    for item in models:
        if not isinstance(item, dict):
            continue
        names = {_normalize(str(item.get("name") or "")),
                 _normalize(str(item.get("model") or ""))}
        if target not in names:
            continue
        details = item.get("details") or {}
        return {
            "found": True,
            "api": api,
            "model": item.get("model") or item.get("name"),
            "digest": item.get("digest"),
            "size": item.get("size"),
            "modified_at": item.get("modified_at"),
            "details": {
                "family": details.get("family"),
                "families": details.get("families"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
                "format": details.get("format"),
                "parent_model": details.get("parent_model"),
            },
        }

    return {
        "found": False,
        "api": api,
        "error": f"未找到模型 {model}",
        "available": sorted(
            {str(m.get("name") or m.get("model")) for m in models if isinstance(m, dict)}
            - {""}
        ),
    }


def model_provenance(
    settings: Any,
    *,
    fetch_json: Optional[FetchJSON] = None,
    timeout_s: float = 2.0,
) -> Dict[str, Any]:
    """把一次调用侧的模型信息整理成 manifest 可落盘的结构。"""
    info = dict(settings.redacted())
    provider = infer_provider(settings.base_url, settings.provider)
    info["provider"] = provider
    if provider == "ollama":
        info["ollama"] = ollama_model_info(
            settings.base_url, settings.model,
            fetch_json=fetch_json, timeout_s=timeout_s,
        )
    return info
