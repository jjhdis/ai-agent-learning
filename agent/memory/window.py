"""
ConversationBufferWindowMemory —— 滑动窗口记忆。

只保留最近 K 轮对话（一轮 = 一次 user + assistant 完整交互），
丢弃更早的对话以控制 Token 消耗。
"""

from agent.memory.base import BaseMemory


class ConversationBufferWindowMemory(BaseMemory):
    """滑动窗口记忆，只保留最近 K 轮对话。

    Parameters:
        k: 保留的最近对话轮数（默认 3）。
           一轮定义为从单条 user 消息开始到 assistant 回复之间的所有消息。
    """

    def __init__(self, k: int = 3):
        self.k = k
        self._history: list[dict] = []

    def load(self) -> list[dict]:
        return list(self._history)

    def save(self, context: list[dict]) -> None:
        self._history = list(context)
        self._trim()

    def clear(self) -> None:
        self._history.clear()

    def _trim(self) -> None:
        """只保留最近 K 轮对话。"""
        user_indices = [
            i for i, m in enumerate(self._history)
            if m.get("role") == "user"
        ]
        if len(user_indices) <= self.k:
            return
        cutoff = user_indices[-(self.k)]
        self._history = self._history[cutoff:]
