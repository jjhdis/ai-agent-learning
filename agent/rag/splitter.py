"""
文档分割器 —— 将长文档切分为可独立检索的小块 (Chunk)。

递归字符分割策略:
    1. 按优先级依次尝试分隔符: 段落(\\n\\n) → 换行(\\n) → 句号(。) → 空格 → 字符
    2. 用当前分隔符切分后，尝试将小片段合并到不超过 chunk_size
    3. 如果单个片段仍超过 chunk_size，降级使用下一级分隔符递归分割
    4. 相邻块之间保留 overlap_size 字符重叠，避免上下文断裂
"""

from agent.rag.document import Document


class RecursiveCharacterTextSplitter:
    """递归字符分割器 —— 按分隔符优先级逐级切分文本。

    Attributes:
        chunk_size: 每块最大字符数
        overlap_size: 相邻块之间的重叠字符数

    使用示例:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, overlap_size=50)
        chunks = splitter.split_documents(docs)  # 或 splitter.split_text(text)
    """

    # 分隔符优先级: 从粗到细
    _SEPARATORS = ["\n\n", "\n", "。", ".", "  ", " "]

    def __init__(self, chunk_size: int = 500, overlap_size: int = 50):
        if chunk_size <= overlap_size:
            raise ValueError(
                f"chunk_size ({chunk_size}) 必须大于 overlap_size ({overlap_size})"
            )
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    # ---- 公开方法 ----

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """将 Document 列表分割为短文档块列表，保留原始元数据。"""
        chunks = []
        for doc in documents:
            for i, text in enumerate(self.split_text(doc.content)):
                meta = dict(doc.metadata)
                meta["chunk_index"] = i
                chunks.append(Document(content=text, metadata=meta))
        return chunks

    def split_text(self, text: str) -> list[str]:
        """将单段文本分割为字符串列表。"""
        return self._split_text(text, list(self._SEPARATORS))

    # ---- 私有 ----

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """递归分割核心。"""
        # 够短就不再切
        stripped = text.strip()
        if len(stripped) <= self.chunk_size:
            return [stripped] if stripped else []

        # 取出当前优先级的分隔符
        sep = separators[0] if separators else ""
        remaining = separators[1:] if len(separators) > 1 else []

        if not sep or sep not in text:
            # 当前分隔符不可用，降级尝试下一级
            if remaining:
                return self._split_text(text, remaining)
            # 无分隔符可用，强制按长度切分
            return self._force_split(text)

        # 按分隔符切分并合并小片段
        parts = text.split(sep)
        result = []
        buffer = ""
        for part in parts:
            merged = buffer + (sep if buffer else "") + part
            if len(merged) <= self.chunk_size:
                buffer = merged
            else:
                if buffer.strip():
                    result.append(buffer)
                if len(part) > self.chunk_size:
                    # 单个片段过长，递归细分
                    result.extend(self._split_text(part, remaining))
                    buffer = ""
                else:
                    buffer = part
        if buffer.strip():
            result.append(buffer)
        return result

    def _force_split(self, text: str) -> list[str]:
        """终极手段: 无视语义，按固定步长切分。"""
        step = max(self.chunk_size - self.overlap_size, 1)
        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks
