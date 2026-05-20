"""
Embedding 客户端 —— 将文本转换为向量表示。

封装 OpenAI 兼容的 Embedding API（text-embedding-3-small / text-embedding-ada-002 等）。
同一套代码可对接 OpenAI、DeepSeek、Qwen 等任何兼容服务。

注意: 如果使用的模型服务商不支持 Embedding API，
可以通过 SimpleEmbeddings 使用本地 TF-IDF 类方案作为降级替代。
"""

import math
import re
from collections import Counter
from openai import OpenAI

from config import EmbeddingConfig


class OpenAIEmbeddings:
    """OpenAI 兼容的 Embedding 客户端。

    默认使用智谱 GLM 的 embedding-2 模型将文本转为向量。

    使用示例:
        emb = OpenAIEmbeddings()
        vec = emb.embed_query("Python 是什么？")
        vecs = emb.embed_documents(["文本1", "文本2", "文本3"])
    """

    def __init__(self, config: EmbeddingConfig = None, model: str = None):
        if config is None:
            from config import Config

            config = Config.embedding
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model = model or config.model

    def embed_query(self, text: str) -> list[float]:
        """将单条查询文本转为向量。"""
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量将文本列表转为向量列表（保持顺序）。"""
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]


class SimpleEmbeddings:
    """本地字符 N-Gram 向量化 —— 无需 API 的降级方案。

    当 Embedding API 不可用时（如使用 DeepSeek 等不支持 Embedding 的服务商），
    使用字符级 bigram 将文本转为稀疏向量，基于 TF 值做余弦相似度检索。

    限制:
        - 语义理解弱于真正的 Embedding 模型
        - 向量维度随语料增长（所有文档的 bigram 并集）
        - 适合小规模学习和演示，生产环境建议使用 OpenAIEmbeddings

    使用示例:
        emb = SimpleEmbeddings()
        vec = emb.embed_query("Python 是什么？")
        vecs = emb.embed_documents(["文本1", "文本2"])
    """

    # 中文常用停用词 + 标点（简化版）
    _STOP_CHARS = set("，。！？、；：""''（）【】《》 \t\n\r!?,.;:\"'()[]{}")

    def __init__(self, ngram: int = 2):
        self._ngram = ngram
        self._vocab: list[str] = []  # 共享词表（仅用于 embed_documents 批量调用）

    def embed_query(self, text: str) -> list[float]:
        """将单条查询文本转为向量。

        如果词表尚未构建（未调用过 embed_documents），
        则从查询文本自身构建临时词表，保证向量非空。
        """
        ngrams = self._extract_ngrams(text)
        if not self._vocab:
            self._vocab = sorted(ngrams.keys())
        return self._ngrams_to_vector(ngrams)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量将文本列表转为向量列表。

        先在全部文档上构建词表，再分别向量化，保证向量维度一致。
        """
        if not texts:
            return []

        # 1. 构建全局词表
        all_ngrams = set()
        doc_ngrams_list = []
        for text in texts:
            ngrams = self._extract_ngrams(text)
            doc_ngrams_list.append(ngrams)
            all_ngrams.update(ngrams.keys())

        self._vocab = sorted(all_ngrams)  # 固定词表保证维度一致

        # 2. 向量化每个文档
        return [self._ngrams_to_vector(ngrams) for ngrams in doc_ngrams_list]

    # ---------- 私有 ----------

    def _extract_ngrams(self, text: str) -> dict[str, int]:
        """提取字符 n-gram 及其词频。"""
        # 预处理: 去停用字符并转小写
        cleaned = "".join(c for c in text if c not in self._STOP_CHARS)
        if not cleaned:
            cleaned = text  # 如果全是标点，保留原文

        counts: dict[str, int] = {}
        n = self._ngram
        for i in range(len(cleaned) - n + 1):
            gram = cleaned[i : i + n]
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    def _ngrams_to_vector(self, ngrams: dict[str, int]) -> list[float]:
        """将 n-gram 词频字典转为词表维度的向量。"""
        return [float(ngrams.get(gram, 0)) for gram in self._vocab]
