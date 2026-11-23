"""agent.rag — 轻量 RAG 包：Ollama embedding + numpy 余弦检索。

设计：
  - 不用 Chroma/Milvus/PGVector，Ollama 自带 /api/embed，
    向量持久化用 numpy (.npy) + 元数据 (.jsonl)，零外部依赖。
  - 对对话修复场景：RAG 提供"历史上相似的坏回复 + 它是怎么被判定的"
    作为 repair prompt 的检索增强，提升判官对长对话的判别力。

面试时能讲清：
  - embedding 生成走 Ollama /api/embeddings
  - 向量库存储：numpy .npy 持久化 + jsonl 元数据
  - 检索：余弦相似度 top-k
  - 接入方式：/repair 加 use_rag + top_k 参数；runtime 加 RAGAugmenter 工具
"""

from .store import VectorStore, DialogueChunk, RAGAugmenter
from .embed import OllamaEmbedder

__all__ = ["VectorStore", "DialogueChunk", "RAGAugmenter", "OllamaEmbedder"]
