"""
RAG (Retrieval-Augmented Generation) 模块。

提供从文档加载到语义检索的完整管道:
    Document → Splitter → Embedding → VectorStore → RetrieverTool

使用示例:
    from agent.rag import RetrieverTool
    tool = RetrieverTool()
    tool.add_texts(["知识文本1", "知识文本2", ...])
    registry.register(tool)  # Agent 即可调用 retrieve_knowledge
"""

from agent.rag.document import Document, TextLoader, TextFileLoader
from agent.rag.splitter import RecursiveCharacterTextSplitter
from agent.rag.embedding import OpenAIEmbeddings, SimpleEmbeddings
from agent.rag.store import InMemoryVectorStore
from agent.rag.retriever import RetrieverTool

__all__ = [
    "Document",
    "TextLoader",
    "TextFileLoader",
    "RecursiveCharacterTextSplitter",
    "OpenAIEmbeddings",
    "SimpleEmbeddings",
    "InMemoryVectorStore",
    "RetrieverTool",
]
