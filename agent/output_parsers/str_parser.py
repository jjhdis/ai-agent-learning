"""
StrOutputParser —— 字符串输出解析器。

最简单的解析器：原样返回 LLM 的文本输出，不做任何转换。
作为默认解析器使用，保证向后兼容。
"""

from typing import Any

from agent.output_parsers.base import BaseOutputParser


class StrOutputParser(BaseOutputParser):
    """字符串输出解析器 —— 最基础的解析器，原样返回文本。

    作为默认解析器使用，确保 Agent.run() 的返回类型向后兼容。
    也可以用作其他解析器的 fallback。

    使用示例:
        parser = StrOutputParser()
        result = parser.parse("这是一段文本")
        print(result)  # "这是一段文本"
    """

    def parse(self, text: str) -> str:
        """原样返回输入文本，不做任何转换。"""
        return text

    def __repr__(self) -> str:
        return "StrOutputParser()"
