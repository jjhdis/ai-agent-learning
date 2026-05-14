"""
ConversationSummaryMemory —— 摘要记忆。

当对话历史过长时，用 LLM 将早期对话压缩成一段摘要，
保留最近几轮的原始消息 + 历史摘要，兼顾上下文和 Token 控制。
"""

from agent.memory.base import BaseMemory
from agent.llm.client import LLMClient


SUMMARY_PROMPT = (
    "请用简洁的中文将以下对话历史压缩成一段摘要（不超过 200 字），"
    "保留关键信息：用户问了什么、Agent 做了什么、最终结果是什么。\n\n"
    "对话历史:\n{history}"
)


class ConversationSummaryMemory(BaseMemory):
    """摘要记忆，用 LLM 压缩早期对话。

    Parameters:
        llm: LLM 客户端，用于生成摘要。
        buffer_size: 保留最近的消息条数（默认 6），其余压缩为摘要。
    """

    def __init__(self, llm: LLMClient, buffer_size: int = 6):
        self.llm = llm
        self.buffer_size = buffer_size
        self._summary: str = ""
        self._history: list[dict] = []

    @property
    def summary(self) -> str:
        return self._summary

    def load(self) -> list[dict]:
        if not self._summary:
            return list(self._history)
        return [{"role": "system", "content": f"[对话历史摘要] {self._summary}"}] + list(self._history)

    def save(self, context: list[dict]) -> None:
        if len(context) > self.buffer_size:
            to_summarize = context[:-self.buffer_size]
            self._history = context[-self.buffer_size:]
            self._summarize(to_summarize)
        else:
            self._history = list(context)

    def clear(self) -> None:
        self._history.clear()
        self._summary = ""

    def _summarize(self, messages: list[dict]) -> None:
        history_text = "\n".join(
            f"[{m['role']}]: {m.get('content', '') or '(调用工具)'}"
            for m in messages
        )
        prompt = SUMMARY_PROMPT.format(history=history_text)
        try:
            msg = self.llm.chat([{"role": "user", "content": prompt}])
            self._summary = msg.content or ""
        except Exception:
            self._summary = f"（共 {len(messages)} 条历史消息，摘要生成失败）"
