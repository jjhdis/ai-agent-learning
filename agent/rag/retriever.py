"""
检索工具 —— 将知识库检索能力封装为 Agent 可调用的 Tool。

RetrieverTool 继承 BaseTool，遵循已有的工具注册和调用协议。
Agent 在 ReAct 循环中可以像调用天气工具一样调用知识检索。

完整 RAG 流程:
    1. 准备文档 → Document
    2. 分割 → RecursiveCharacterTextSplitter.split_documents()
    3. 向量化 + 存储 → InMemoryVectorStore.add_documents()
    4. 检索 → RetrieverTool.execute(query)
"""

from agent.tools.base import BaseTool, ToolParameter
from agent.rag.store import InMemoryVectorStore
from agent.rag.document import Document
from agent.rag.splitter import RecursiveCharacterTextSplitter


class RetrieverTool(BaseTool):
    """知识库检索工具 —— 让 Agent 能从外部文档中查找信息。

    使用示例:
        tool = RetrieverTool()
        tool.add_texts([
            "Python 由 Guido van Rossum 于 1991 年发布...",
            "Python 的 GIL (Global Interpreter Lock) 是...",
        ])
        registry.register(tool)  # 注册后 Agent 即可调用
    """

    name = "retrieve_knowledge"
    description = (
        "从知识库中检索与查询相关的信息。"
        "当需要查找事实、背景知识或文档中的特定信息时使用此工具。"
        "参数 query 应该是具体的搜索关键词或问题。"
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="搜索查询词或具体问题，用于从知识库中检索相关信息",
        ),
    ]

    def __init__(
        self,
        vector_store: InMemoryVectorStore = None,
        k: int = 4,
        chunk_size: int = 500,
        overlap_size: int = 50,
    ):
        self._store = vector_store or InMemoryVectorStore()
        self._k = k
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, overlap_size=overlap_size
        )

    # ---- 属性 ----

    @property
    def store(self) -> InMemoryVectorStore:
        """获取底层向量存储，用于添加文档或查看存储状态。"""
        return self._store

    @property
    def document_count(self) -> int:
        """知识库中文档块的总数。"""
        return self._store.document_count

    # ---- 添加文档 ----

    def add_texts(
        self, texts: list[str], metadatas: list[dict] = None
    ) -> None:
        """添加原始文本到知识库（自动分割 + 向量化 + 存储）。

        Args:
            texts: 原始文本列表
            metadatas: 可选的元数据列表，与 texts 一一对应
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        docs = []
        for i, text in enumerate(texts):
            meta = {"index": i}
            if i < len(metadatas):
                meta.update(metadatas[i])
            docs.append(Document(content=text, metadata=meta))

        self.add_documents(docs)

    def add_documents(self, documents: list[Document]) -> None:
        """添加 Document 列表到知识库（自动分割 + 向量化 + 存储）。"""
        chunks = self._splitter.split_documents(documents)
        self._store.add_documents(chunks)

    def load_and_split(self, file_path: str) -> None:
        """加载文本文件，分割后存入知识库。

        一站式方法: 读取文件 → 分块 → 向量化 → 存储。
        """
        from agent.rag.document import TextFileLoader

        docs = TextFileLoader(file_path).load()
        self.add_documents(docs)

    # ---- 核心: 执行检索 ----

    def execute(self, query: str) -> str:
        """执行知识检索 —— Agent 通过 function calling 触发此方法。

        Args:
            query: 搜索查询词

        Returns:
            格式化的检索结果字符串，包含来源和内容摘要
        """
        if self._store.document_count == 0:
            return "[知识库为空] 尚未添加任何文档到知识库中。"

        docs = self._store.similarity_search(query, k=self._k)

        if not docs:
            return "未找到与查询相关的信息。"

        lines = [f"知识库检索结果（共匹配 {len(docs)} 条）:\n"]
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            chunk_idx = doc.metadata.get("chunk_index", "")
            label = f"[{i}] 来源: {source}" + (
                f", 片段 #{chunk_idx}" if chunk_idx != "" else ""
            )
            lines.append(label)

            # 截断过长的内容，保留关键信息
            content = doc.content.strip()
            if len(content) > 400:
                content = content[:400] + "…"
            lines.append(f"    {content}")
            lines.append("")

        return "\n".join(lines)
