"""配置与密钥读取

两条原则：
1. 密钥只从环境变量读，不落盘、不进日志、不进快照
2. .env 只是"把值放进环境变量"的便捷方式，代码不解析业务配置

不引入 python-dotenv：解析逻辑不到 20 行，少一个依赖。
"""

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

_ENV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"


class MissingCredentials(RuntimeError):
    """缺少 API 密钥。宁可启动即失败，也不要带着空 key 去请求。"""


def _strip_inline_comment(val: str) -> str:
    """去掉未加引号值末尾的 "# ..." 注释；引号内的 # 保留。

    约定：只有在空白之后出现的 # 才算注释起点，
    避免把 URL fragment 之外的合法值误伤（如颜色值 #fff 不加空白时保留）。
    """
    if not val:
        return val
    # KEY=   # 注释 → 正则吃掉空白后以 "# " 开头，等价于留空；
    # 而 "#fff" 这种紧跟非空白的值仍按值处理
    if val.startswith("#"):
        return "" if len(val) == 1 or val[1].isspace() else val
    if val[0] in "\"'":
        end = val.find(val[0], 1)
        if end != -1:
            return val[1:end]
    m = re.search(r"\s+#", val)
    if m:
        val = val[:m.start()]
    return val.strip()


def parse_dotenv(text: str) -> Dict[str, str]:
    """解析 KEY=VALUE 行；忽略注释与空行，允许单双引号包裹与行尾注释。"""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), _strip_inline_comment(m.group(2).strip())
        out[key] = val
    return out


def load_dotenv(
    path: Union[str, Path, None] = ".env",
    *,
    override: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """把 .env 载入环境变量。

    默认不覆盖已存在的环境变量：CI/生产注入的值优先于本地文件。
    文件不存在时静默返回空字典。
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    values = parse_dotenv(p.read_text(encoding="utf-8"))
    env = os.environ if environ is None else environ
    for k, v in values.items():
        if override or not env.get(k):
            os.environ[k] = v
    return values


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "deepseek"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_KEY_ENV
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 60.0
    json_mode: bool = True

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "LLMSettings":
        env = os.environ if environ is None else environ
        base = cls()
        kwargs: Dict[str, Any] = {}
        kwargs["model"] = env.get("DEEPSEEK_MODEL") or base.model
        kwargs["base_url"] = env.get("DEEPSEEK_BASE_URL") or base.base_url

        if raw := env.get("DEEPSEEK_TEMPERATURE"):
            try:
                kwargs["temperature"] = float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"DEEPSEEK_TEMPERATURE={raw!r} 不是合法浮点数"
                )
        if raw := env.get("DEEPSEEK_MAX_TOKENS"):
            try:
                kwargs["max_tokens"] = int(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"DEEPSEEK_MAX_TOKENS={raw!r} 不是合法整数"
                )
        if raw := env.get("DEEPSEEK_TIMEOUT"):
            try:
                kwargs["timeout_s"] = float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"DEEPSEEK_TIMEOUT={raw!r} 不是合法浮点数"
                )
        if raw := env.get("DEEPSEEK_JSON_MODE"):
            low = raw.strip().lower()
            if low in ("true", "1", "yes", "on"):
                kwargs["json_mode"] = True
            elif low in ("false", "0", "no", "off"):
                kwargs["json_mode"] = False
            else:
                raise ValueError(
                    f"DEEPSEEK_JSON_MODE={raw!r} 无法解析为布尔值"
                    "（合法值：true/false/1/0/yes/no/on/off）"
                )

        return replace(base, **kwargs)

    def api_key(self, environ: Optional[Mapping[str, str]] = None) -> str:
        env = os.environ if environ is None else environ
        key = (env.get(self.api_key_env) or "").strip()
        if not key:
            raise MissingCredentials(
                f"未设置 {self.api_key_env}：请写入 .env 或导出环境变量"
            )
        return key

    def has_key(self, environ: Optional[Mapping[str, str]] = None) -> bool:
        env = os.environ if environ is None else environ
        return bool((env.get(self.api_key_env) or "").strip())

    def redacted(self) -> dict:
        """可安全写进日志/元数据的配置视图（不含密钥）。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "json_mode": self.json_mode,
            "has_key": self.has_key(),
        }


def judge_settings(environ: Optional[Mapping[str, str]] = None) -> LLMSettings:
    """判官侧配置：JUDGE_MODEL / JUDGE_BASE_URL 可独立覆盖生成侧。

    所有 JUDGE_* 环境变量留空表示"沿用 DEEPSEEK_* 对应值"。
    判官必须确定性，强制 temperature=0.0。
    """
    env = os.environ if environ is None else environ
    base = LLMSettings.from_env(env)

    kwargs: Dict[str, Any] = {"temperature": 0.0}
    if m := env.get("JUDGE_MODEL", "").strip():
        kwargs["model"] = m
    if u := env.get("JUDGE_BASE_URL", "").strip():
        kwargs["base_url"] = u

    return replace(base, **kwargs)
