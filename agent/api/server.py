"""FastAPI 服务化：把对话修复 Agent 的决策循环暴露成 HTTP。

运行方式：
    python -m agent.api                          # 默认 0.0.0.0:8000
    DEEPSEEK_BASE_URL=http://127.0.0.1:11434/v1 \
    DEEPSEEK_API_KEY=dummy \
    JUDGE_MODEL=qwen3-coder-30b:latest \
    python -m agent.api --host 0.0.0.0 --port 8000

依赖：pip install "dialogue-repair-agent[api,llm]"
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import httpx
except ImportError as e:  # FastAPI / pydantic 是可选依赖
    missing = str(e).split()[-1]
    raise SystemExit(
        f"缺少可选依赖：pip install 'dialogue-repair-agent[api,llm]'"
    ) from e

from ..benchmark.agent_eval import (
    ARMS,
    build_runtime,
)
from ..eval.judge import JUDGE_VERSION, LLMJudge, RUBRIC_VERSION
from ..runtime.config import LLMSettings, judge_settings, load_dotenv

# --------------------------------------------------------------------------- 全局
# lifespan 里初始化一次 client（进程内缓存，连接池复用）——每个请求只是 build_runtime，
# 不新建底层 httpx.Client。
_client: Any = None
_settings: Optional[LLMSettings] = None
_stamp: str = ""


# --------------------------------------------------------------------------- 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载 env + 建 client；关闭时释放连接池。"""
    global _client, _settings, _stamp
    load_dotenv()
    _settings = LLMSettings.from_env()
    _stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # build_stage_clients 走 sync 代码路径，放到 asyncio.to_thread 避免阻塞 event loop
    from ..runtime.deepseek import build_stage_clients
    _client = await asyncio.to_thread(build_stage_clients, ledger=None)

    try:
        yield
    finally:
        # 释放 httpx.Client
        for attr in ("_client", "inner", "inner_inner"):
            c = getattr(_client, attr, None)
            if hasattr(c, "close"):
                try:
                    c.close()
                except Exception:
                    pass
        _client = None


app = FastAPI(
    title="Dialogue Repair Agent",
    version="0.1.0",
    description=(
        "LLM Agent 修复模板拼接对话的跨轮断裂。"
        "暴露 A4 / A4R 等决策循环为 HTTP 端点。"
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- 请求/响应模型
class DialogueMessage(BaseModel):
    role: str = Field(..., pattern=r"^(system|user|assistant)$")
    content: str


class RepairRequest(BaseModel):
    dialogue: Dict[str, Any] = Field(
        ...,
        description="待校验/修复的对话，形如 {messages: [{role, content}, ...]}",
        examples=[{"messages": [
            {"role": "system", "content": "你是客服助手。"},
            {"role": "user", "content": "【通话开始】"},
            {"role": "assistant", "content": "您好，请问是张三吗？"},
            {"role": "user", "content": "对。"},
            {"role": "assistant", "content": "哦对了您的借款今天到期了。"},
            {"role": "user", "content": "什么借款？"},
        ]}],
    )
    arm: str = Field(default="A4", description=f"决策臂：{list(ARMS)}")
    reference: Optional[Dict[str, Any]] = Field(
        default=None,
        description="回退参照（原始对话）；缺省时用 dialogue 自身为基准，一致性校验必然通过",
    )
    ignore_turns: List[int] = Field(
        default_factory=list,
        description="判官需忽略的 assistant turn 列表（私有轨业务伪影屏蔽用）",
        examples=[[2, 4]],
    )
    run_id: Optional[str] = Field(
        default=None,
        description="可选 run_id；缺省按时间戳自动生成",
    )


class RepairResponse(BaseModel):
    run_id: str
    arm: str
    status: str
    termination: str
    reason: str
    iterations: int
    tool_calls: int
    tokens: int
    ok: bool
    issues: List[Dict[str, Any]]
    dialogue: Dict[str, Any]


class ArmInfo(BaseModel):
    name: str
    description: str


ARMLY_DESCRIPTIONS: Dict[str, str] = {
    "A2": "单次 LLM 重写（无校验，作为上限 baseline）",
    "A3": "纯规则 + 确定性工具（零 LLM 调用，offline 臂）",
    "A3L": "规则 + LLM function calling + 确定性工具",
    "A3R": "A3 + rewrite_segment 局部改写",
    "A4": "LLM 决策循环 + 语义判官 + 确定性工具",
    "A4R": "A4 + rewrite_segment（完整闭环）",
}


# --------------------------------------------------------------------------- 端点
@app.get("/health")
async def health() -> Dict[str, Any]:
    """服务 + LLM 健康检查。"""
    settings = _settings
    assert settings is not None, "lifespan 未初始化"
    info: Dict[str, Any] = {
        "status": "ok",
        "agent_version": "0.1.0",
        "run_id_stamp": _stamp,
        "judge_version": JUDGE_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "supports_arms": list(ARMS),
        "llm": {
            "provider": settings.provider,
            "model": settings.model,
            "base_url": settings.base_url,
            "temperature": settings.temperature,
            "has_key": settings.has_key(),
        },
    }
    # 可达性检查：Ollama 暴露 /api/tags，DeepSeek 直接 /v1/models
    try:
        async with httpx.AsyncClient(timeout=3.0) as hc:
            url = settings.base_url.rstrip("/")
            if "ollama" in url.lower() or ":11434" in url:
                r = await hc.get("http://127.0.0.1:11434/api/tags")
                info["llm"]["ollama_models"] = [
                    m["name"] for m in (r.json().get("models") or [])
                ]
            else:
                r = await hc.get(f"{url}/models")
            info["llm"]["reachable"] = True
            info["llm"]["http"] = r.status_code
    except Exception as e:
        info["llm"]["reachable"] = False
        info["llm"]["error"] = str(e)
    return info


@app.get("/api/v1/arms", response_model=List[ArmInfo])
async def list_arms() -> List[ArmInfo]:
    """列出可用的决策臂。"""
    return [ArmInfo(name=a, description=ARMLY_DESCRIPTIONS.get(a, "")) for a in ARMS]


@app.post("/api/v1/repair", response_model=RepairResponse)
async def repair(req: RepairRequest) -> RepairResponse:
    """跑一轮修复决策循环。"""
    if req.arm not in ARMS:
        raise HTTPException(400, f"未知 arm：{req.arm}（可选 {list(ARMS)}）")
    messages = req.dialogue.get("messages")
    if not messages:
        raise HTTPException(400, "dialogue 缺少 messages 列表")

    run_id = req.run_id or f"api_{time.strftime('%H%M%S')}"
    ignore_fs = frozenset(req.ignore_turns) if req.ignore_turns else frozenset()

    # A4/A4R 必须带判官；A3 系列不需要（纯规则/LLM function calling 用不到 JudgeError 契约）
    judge = None
    if req.arm in ("A4", "A4R"):
        judge = LLMJudge(client=_client, run_id=f"judge_{run_id}", ignore_turns=ignore_fs)

    runtime = build_runtime(req.arm, client=_client, judge=judge, run_id=run_id)
    try:
        result = runtime.run(req.dialogue, reference=req.reference)
    except Exception as e:
        raise HTTPException(
            502,
            f"决策循环失败：{type(e).__name__}: {str(e)[:300]}",
        ) from e

    return RepairResponse(
        run_id=run_id,
        arm=req.arm,
        status=result.status,
        termination=result.termination,
        reason=result.reason,
        iterations=result.iterations,
        tool_calls=result.tool_calls,
        tokens=result.tokens,
        ok=result.ok,
        issues=result.issues + result.seen_issues,
        dialogue=result.dialogue,
    )


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    """兜底：任何未捕获异常都走 500，不泄露 traceback 细节。"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)[:200]}"},
    )
