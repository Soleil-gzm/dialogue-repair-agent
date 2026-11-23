"""OllamaEmbedder — 走 Ollama /api/embeddings 的文本向量化。

接口：embed(text: str) -> list[float]
接口：embed_batch(texts: list[str]) -> list[list[float]]
接口：dim() -> int

完全零额外依赖：只需要 httpx（项目里已经装了）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass
class OllamaEmbedder:
    model: str = "bge-m3:latest"
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 10.0

    def __post_init__(self):
        self._dim: Optional[int] = None

    def _http(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout_s)

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Ollama /api/embeddings 一次只接受单条 prompt，for 循环串行
        results: List[List[float]] = []
        with self._http() as hc:
            for t in texts:
                r = hc.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                )
                r.raise_for_status()
                data = r.json()
                emb = data.get("embedding") or data.get("embeddings") or []
                if not emb:
                    raise RuntimeError(f"Ollama embed 无向量返回（model={self.model}）")
                self._dim = len(emb[0]) if isinstance(emb[0], list) else len(emb)
                # Ollama 单条返回是 flat array，batch 返回是 [[...],[...]]
                if isinstance(emb[0], list):
                    results.append(emb[0])
                else:
                    results.append(emb)
        return results

    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dim probe"))
        return self._dim
