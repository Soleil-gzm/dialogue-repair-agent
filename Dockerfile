# =============================================================
# Dialogue Repair Agent — API 容器（多阶段构建）
# =============================================================
# 构建：   docker build -t dialogue-repair-agent:latest .
# 运行：   docker run --rm -p 8000:8000 \
#                      -e DEEPSEEK_BASE_URL=http://host.docker.internal:11434/v1 \
#                      -e DEEPSEEK_API_KEY=dummy \
#                      -e JUDGE_MODEL=qwen3-coder-30b:latest \
#                      dialogue-repair-agent:latest
# Compose：docker compose up --build
# =============================================================

# ---- 阶段 1：builder（装依赖，体积大，构建完扔掉）----
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir --upgrade pip

# 先只复制依赖声明文件——利用 Docker 层缓存：依赖不变时重建不重新 pip install
WORKDIR /app
COPY pyproject.toml README.md ./

# 装运行时需要的所有包：FastAPI + uvicorn + httpx + tenacity
# setuptools 会自动根据 pyproject.toml 的 packages 登记把 agent.* 装进去
RUN pip install --no-cache-dir ".[api,llm]"

# ---- 阶段 2：runtime（瘦镜像，只复制装好依赖的 venv + 源码）----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# 创建 venv 目录，从 builder 复制已装好的包
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder /usr/local/lib/python3.12/site-packages /opt/venv/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /opt/venv/bin
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 复制源码
COPY agent ./agent
COPY pyproject.toml README.md ./

# 默认 env（compose / docker run -e 会覆盖）
ENV DEEPSEEK_BASE_URL="http://ollama:11434/v1" \
    DEEPSEEK_API_KEY="ollama-local" \
    DEEPSEEK_MODEL="qwen3:8b" \
    JUDGE_MODEL="qwen3-coder-30b:latest" \
    DEEPSEEK_TEMPERATURE=0.0 \
    DEEPSEEK_MAX_TOKENS=2048

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info", "--no-access-log"]
