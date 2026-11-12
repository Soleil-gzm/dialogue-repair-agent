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
from typing import Dict, Mapping, Optional, Union

_ENV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"


class MissingCredentials(RuntimeError):
    """缺少 API 密钥。宁可启动即失败，也不要带着空 key 去请求。"""


def parse_dotenv(text: str) -> Dict[str, str]:
    """解析 KEY=VALUE 行；忽略注释与空行，允许单双引号包裹。"""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
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
        return replace(
            base,
            model=env.get("DEEPSEEK_MODEL") or base.model,
            base_url=env.get("DEEPSEEK_BASE_URL") or base.base_url,
        )

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
