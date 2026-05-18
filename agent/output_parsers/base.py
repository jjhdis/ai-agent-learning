"""
BaseOutputParser —— 输出解析器抽象基类。

定义解析器的统一接口，所有具体解析器都从此类继承。
同时提供 OutputParserException 用于解析失败时的错误传递。
"""

from typing import Any

from agent.chain.runnable import Runnable


class OutputParserException(ValueError):
    """输出解析异常，携带原始文本和解析目标类型信息。"""

    def __init__(self, message: str, text: str = "", parser_type: str = ""):
        super().__init__(message)
        self.text = text
        self.parser_type = parser_type

    def __str__(self) -> str:
        base = super().__str__()
        if self.parser_type:
            base = f"[{self.parser_type}] {base}"
        return base


class BaseOutputParser(Runnable):
    """输出解析器抽象基类。

    将 LLM 的非结构化文本输出转换为结构化数据。
    所有解析器都需实现 parse() 方法。
    可选实现 get_format_instructions() 方法来生成格式引导提示词，
    将该说明追加到 System Prompt 中可引导 LLM 按指定格式输出。

    Parameters:
        skip_empty: 是否跳过空输入直接返回，默认 False

    使用示例:
        parser = JsonOutputParser()
        result = parser.parse('{"name": "上海", "temp": 28}')
        # 或在管道中使用:
        # chain = prompt | llm | parser
    """

    def __init__(self, skip_empty: bool = False):
        self.skip_empty = skip_empty

    def parse(self, text: str) -> Any:
        """将 LLM 文本输出解析为目标数据结构。

        Args:
            text: LLM 返回的原始文本

        Returns:
            解析后的结构化数据

        Raises:
            OutputParserException: 解析失败
        """
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        """返回格式说明文本，可追加到 System Prompt 中引导 LLM 按格式输出。

        Returns:
            格式说明字符串，默认返回空字符串
        """
        return ""

    def invoke(self, input_data: Any) -> Any:
        """Runnable 协议接口。

        支持多种输入格式:
        - 字符串: 直接解析
        - 字典: 取 'content' / 'text' 键的值解析
        - 对象: 取 .content 属性解析
        - None / 空字符串: skip_empty=True 时返回 None
        """
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, dict):
            text = input_data.get("content") or input_data.get("text") or ""
            if not text and isinstance(input_data, dict) and len(input_data) == 1:
                # 单键字典，取唯一值作为文本
                text = str(next(iter(input_data.values())))
            elif not text:
                text = str(input_data)
        elif hasattr(input_data, "content"):
            text = getattr(input_data, "content", "")
        elif input_data is None:
            text = ""
        else:
            text = str(input_data)

        if self.skip_empty and not text.strip():
            return None

        return self.parse(text)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
