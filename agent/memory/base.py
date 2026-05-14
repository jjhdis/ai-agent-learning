"""
Memory 基类 —— 为 Agent 提供跨轮次对话记忆能力。

所有 Memory 实现只需继承此基类并实现 load / save / clear 方法。
"""

from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """对话记忆的抽象基类。

    负责管理对话历史的持久化与策略（全量/窗口/摘要等），
    让 Agent 在多次 run() 调用之间保持上下文。
    """

    @abstractmethod
    def load(self) -> list[dict]:
        """加载当前记忆中的对话历史。"""
        ...

    @abstractmethod
    def save(self, context: list[dict]) -> None:
        """将完整对话历史保存到记忆中。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空记忆。"""
        ...
