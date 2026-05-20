"""
向量存储 —— 将文档向量化并支持语义相似度检索。

InMemoryVectorStore:
    内存向量存储，使用余弦相似度检索。
    适合学习和小规模知识库（< 1 万条）。

设计理念:
    - 先向量化后存储: add_documents() 调用 embeddings API
    - 检索时向量化查询: similarity_search() 调用 embeddings API
    - 向量与文档通过列表索引关联，一一对应
"""

import math

from agent.rag.document import Document
from agent.rag.embedding import OpenAIEmbeddings


class InMemoryVectorStore:
    """内存向量存储 —— 余弦相似度检索。

    使用示例:
        store = InMemoryVectorStore()
        store.add_documents(chunks)                # 向量化并存入
        results = store.similarity_search("查询", k=4)  # 检索 top-4
    """

    def __init__(self, embedding: OpenAIEmbeddings = None):
        self._embedding = embedding or OpenAIEmbeddings()
        self._documents: list[Document] = []
        self._vectors: list[list[float]] = []

    # ---- 属性 ----

    @property
    def document_count(self) -> int:
        """已存储文档数量。"""
        return len(self._documents)

    # ---- 写入 ----

    def add_documents(self, documents: list[Document]) -> None:
        """向量化文档并追加到存储中。"""
        if not documents:
            return
        texts = [doc.content for doc in documents]
        vectors = self._embedding.embed_documents(texts)
        self._documents.extend(documents)
        self._vectors.extend(vectors)

    # ---- 检索 ----

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """余弦相似度检索 —— 返回与查询最相关的 k 个文档。

        Returns:
            按相似度降序排列的 Document 列表（最多 k 个）
        """
        if not self._documents:
            return []

        query_vec = self._embedding.embed_query(query)
        scores = []
        for i, vec in enumerate(self._vectors):
            score = self._cosine_similarity(query_vec, vec)
            scores.append((score, i))

        # 按相似度降序
        scores.sort(key=lambda x: x[0], reverse=True)

        # 返回 top-k
        top_k = min(k, len(scores))
        return [self._documents[i] for _, i in scores[:top_k]]

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """同 similarity_search，额外返回相似度分数。"""
        if not self._documents:
            return []

        query_vec = self._embedding.embed_query(query)
        scores = []
        for i, vec in enumerate(self._vectors):
            score = self._cosine_similarity(query_vec, vec)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = min(k, len(scores))
        return [(self._documents[i], score) for score, i in scores[:top_k]]

    def clear(self) -> None:
        """清空所有已存储的文档和向量。"""
        self._documents.clear()
        self._vectors.clear()

    # ---- 静态 ----

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """两个等长向量的余弦相似度，值域 [-1, 1]，越大越相似。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
