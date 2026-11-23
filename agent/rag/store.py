"""VectorStore — numpy 余弦检索的持久化向量库。

设计选择：
  - 不用 Chroma/Milvus/PGVector，因为 JD 里"向量数据库"的面试目标是
    证明你理解 embedding → 存储 → 检索这条链路。numpy + jsonl 足够。
  - 持久化：vectors.npy (numpy float32) + metadata.jsonl (一行一条元数据)
  - 检索：余弦相似度 top-k，numpy.dot 实现。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .embed import OllamaEmbedder


@dataclass
class DialogueChunk:
    """要被向量化 + 存库的片段——对对话修复场景，一条 chunk = 一轮问答。"""
    dialogue_id: str
    user_turn: int
    assistant_turn: int
    user_content: str
    assistant_content: str
    # 元数据：缺陷类型（若有）、是否答非所问、修复前后（若已修复）
    defect_type: str = ""
    is_off_topic: bool = False
    repair_before: str = ""
    repair_after: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """拼接成一段待 embedding 的文本。"""
        return f"user: {self.user_content}\nassistant: {self.assistant_content}"


class VectorStore:
    """numpy 后端的持久化向量库。"""

    def __init__(self, path: Path | str, dim: int = 1024):
        self.path = Path(path)
        self.vec_path = self.path / "vectors.npy"
        self.md_path = self.path / "metadata.jsonl"
        self.dim = dim
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._meta: List[Dict[str, Any]] = []

    # ---------------- 持久化 ----------------
    def load(self) -> None:
        if self.vec_path.exists():
            self._vectors = np.load(self.vec_path)
        if self.md_path.exists():
            self._meta = [json.loads(l) for l in self.md_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.vec_path, self._vectors)
        with self.md_path.open("w", encoding="utf-8") as f:
            for m in self._meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---------------- 索引 ----------------
    @property
    def size(self) -> int:
        return len(self._meta)

    def add(self, embeddings: List[List[float]], meta: List[Dict[str, Any]]) -> None:
        if not embeddings:
            return
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._vectors = np.concatenate([self._vectors, arr]) if self._vectors.size else arr
        self._meta.extend(meta)

    def add_chunk(self, chunk: DialogueChunk, embedder: OllamaEmbedder) -> None:
        emb = embedder.embed(chunk.text)
        self.add([emb], [asdict(chunk)])

    def add_chunks(self, chunks: List[DialogueChunk], embedder: OllamaEmbedder, batch: int = 16) -> None:
        texts = [c.text for c in chunks]
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i + batch]
            batch_meta = [asdict(c) for c in chunks[i:i + batch]]
            embs = embedder.embed_batch(batch_texts)
            self.add(embs, batch_meta)

    # ---------------- 检索 ----------------
    def search(self, query_embedding: List[float], top_k: int = 5,
               filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """余弦相似度 top-k 检索。

        filter: 可选字典，如 {"defect_type": "D1_off_topic"} 只返回匹配元数据。
        """
        if self._vectors.size == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        v_norm = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-9)
        sims = v_norm @ q_norm  # [n]

        # 过滤
        idxs = np.arange(len(self._meta))
        if filter:
            mask = np.array([
                all(self._meta[i].get(k) == v for k, v in filter.items())
                for i in range(len(self._meta))
            ])
            idxs = idxs[mask]
            sims = sims[mask]

        if len(idxs) == 0:
            return []
        # top-k
        k = min(top_k, len(idxs))
        top_pos = np.argsort(-sims)[:k]
        results = []
        for p in top_pos:
            orig_idx = int(idxs[p])
            results.append({"score": float(sims[p]), "meta": self._meta[orig_idx]})
        return results


# =====================================================================
# RAGAugmenter — 接到 Agent runtime 的增强组件
# 对对话修复场景：RAG 提供"历史上相似的坏回复 + 它是怎么被判定/修复的"
# 作为 repair prompt 的上下文增强
# =====================================================================

class RAGAugmenter:
    """对话修复专用的 RAG 组件。

    用法 1（独立查）：augmenter.query(current_pair, top_k=5) → 历史相似坏回复
    用法 2（接到 runtime）：作为 rewrite_segment 的上下文增强——把 top-k 历史案例
      拼到 rewrite 的 prompt 里
    """

    def __init__(self, store: VectorStore, embedder: OllamaEmbedder):
        self.store = store
        self.embedder = embedder

    def query(self, user_content: str, assistant_content: str,
              top_k: int = 5) -> List[Dict[str, Any]]:
        emb = self.embedder.embed(f"user: {user_content}\nassistant: {assistant_content}")
        return self.store.search(emb, top_k=top_k)

    def render_for_prompt(self, hits: List[Dict[str, Any]]) -> str:
        """把 top-k hits 渲染成一段可以拼进 repair prompt 的文本。"""
        if not hits:
            return ""
        lines = ["### 历史相似案例（RAG 检索）"]
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            label = f"[{m.get('defect_type','')}={'答非所问' if m.get('is_off_topic') else '正常'}]"
            lines.append(
                f"{i}. score={h['score']:.2f} {label}\n"
                f"   user: {m.get('user_content','')[:80]}\n"
                f"   assistant(before): {m.get('assistant_content','')[:80]}\n"
                f"   assistant(after):   {(m.get('repair_after') or '(未修复)')[:80]}"
            )
        return "\n".join(lines)
