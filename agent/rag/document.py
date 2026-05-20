"""
文档模型与加载器。

Document 是 RAG 系统中最基本的数据单元，代表一段文本及其元数据。
加载器负责从不同来源（文件、字符串列表等）读取内容并生成 Document 列表。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """文档数据模型 —— RAG 管道中的基本数据单元。

    Attributes:
        content: 文档文本内容
        metadata: 附加元数据（来源、页码、标题等），检索时可追溯原文出处
    """

    content: str
    metadata: dict = field(default_factory=dict)


class TextFileLoader:
    """从文本文件加载文档。

    支持 .txt / .md / .py 等纯文本文件。
    未来可扩展支持 PDF、Word、HTML 等格式。

    使用示例:
        loader = TextFileLoader("knowledge.txt")
        docs = loader.load()
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self._file_path = Path(file_path)
        self._encoding = encoding

    def load(self) -> list[Document]:
        """读取文件内容并返回 Document 列表。"""
        if not self._file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self._file_path}")
        with open(self._file_path, "r", encoding=self._encoding) as f:
            content = f.read()
        return [Document(content=content, metadata={"source": str(self._file_path)})]


class TextLoader:
    """从字符串列表加载文档。

    快速构建测试知识库，无需创建文件。

    使用示例:
        loader = TextLoader([
            "Python 是一种解释型编程语言...",
            "Python 的 GIL 机制限制了多线程...",
        ])
        docs = loader.load()
    """

    def __init__(self, texts: list[str], metadatas: list[dict] = None):
        self._texts = texts
        self._metadatas = metadatas or [{}] * len(texts)

    def load(self) -> list[Document]:
        """将字符串列表转为 Document 列表。"""
        docs = []
        for i, text in enumerate(self._texts):
            meta = {"index": i}
            if i < len(self._metadatas):
                meta.update(self._metadatas[i])
            docs.append(Document(content=text, metadata=meta))
        return docs
