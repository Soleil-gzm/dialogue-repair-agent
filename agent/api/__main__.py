"""`python -m agent.api` — 起一个 FastAPI server。

用法：
    python -m agent.api
    python -m agent.api --host 127.0.0.1 --port 8000
    python -m agent.api --reload           # 开发模式热重载（需安装 uvicorn[standard]）
"""

from __future__ import annotations

import argparse
import os
import sys


def _missing(pkg: str) -> SystemExit:
    return SystemExit(
        f"缺少可选依赖 {pkg}：pip install 'dialogue-repair-agent[api,llm]'"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对话修复 Agent HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="热重载（开发用）")
    args = parser.parse_args(argv)

    # 提前探测依赖——比 uvicorn 启动时炸掉给的错误信息友好
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        raise _missing("uvicorn")
    try:
        import fastapi  # noqa: F401
    except ImportError:
        raise _missing("fastapi")

    # 提前探 LLM 配置——没配也能起，/repair 会明确报错
    os.environ.setdefault("DEEPSEEK_BASE_URL", "http://127.0.0.1:11434/v1")
    os.environ.setdefault("DEEPSEEK_API_KEY", "ollama-local")
    os.environ.setdefault("DEEPSEEK_MODEL", "qwen3:8b")
    os.environ.setdefault("JUDGE_MODEL", "qwen3-coder-30b:latest")

    import uvicorn
    uvicorn.run(
        "agent.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
