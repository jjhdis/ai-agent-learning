"""
ConversationBufferMemory —— 完整保存所有对话历史。

最简单的记忆策略：将每轮对话的消息原样存储。
可设置 max_messages 限制总消息数，防止 Token 溢出。
"""

from agent.memory.base import BaseMemory


class ConversationBufferMemory(BaseMemory):
    """完整对话记忆。

    Parameters:
        max_messages: 最大消息条数，超出时自动丢弃最早的消息。
                      设为 0 表示不限制。
    """

    def __init__(self, max_messages: int = 0):
        self._history: list[dict] = []
        self.max_messages = max_messages

    def load(self) -> list[dict]:
        return list(self._history)

    def save(self, context: list[dict]) -> None:
        self._history = list(context)
        if self.max_messages > 0 and len(self._history) > self.max_messages:
            self._history = self._history[-self.max_messages:]

    def clear(self) -> None:
        self._history.clear()
