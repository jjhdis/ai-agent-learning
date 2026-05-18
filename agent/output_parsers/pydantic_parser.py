"""
PydanticOutputParser —— Pydantic 模型输出解析器。

将 LLM 的 JSON 输出解析为指定的 Pydantic 模型实例。
自动从模型定义生成 JSON Schema 格式说明，引导 LLM 按结构输出。
"""

import json
import re
from typing import Any, Type

from agent.output_parsers.base import BaseOutputParser, OutputParserException


class PydanticOutputParser(BaseOutputParser):
    """Pydantic 输出解析器。

    将 LLM 的 JSON 输出解析为指定的 Pydantic BaseModel 实例。
    自动从模型字段生成格式说明（含字段名、类型、描述、必填/可选），
    将该说明追加到 System Prompt 可引导 LLM 严格按结构输出。

    Parameters:
        pydantic_object: Pydantic BaseModel 子类

    Raises:
        ImportError: 未安装 pydantic 时抛出
        OutputParserException: 解析或验证失败时抛出

    使用示例:
        from pydantic import BaseModel, Field

        class WeatherInfo(BaseModel):
            city: str = Field(description="城市名称")
            temperature: float = Field(description="温度（摄氏度）")
            condition: str = Field(description="天气状况，如晴/多云/雨")

        parser = PydanticOutputParser(pydantic_object=WeatherInfo)
        instructions = parser.get_format_instructions()
        # 将 instructions 追加到 System Prompt...

        result = parser.parse('{"city": "上海", "temperature": 28.0, "condition": "晴"}')
        print(result.city)           # "上海"
        print(type(result))          # <class 'WeatherInfo'>
    """

    def __init__(self, pydantic_object: Type):
        super().__init__()
        self.pydantic_object = pydantic_object
        self._validate_pydantic()

    def parse(self, text: str) -> Any:
        """将 JSON 文本解析为 Pydantic 模型实例。

        Raises:
            OutputParserException: JSON 解析失败或 Pydantic 验证失败
        """
        text = text.strip()
        if not text:
            raise OutputParserException(
                "输入文本为空，无法解析为 Pydantic 模型。",
                text=text,
                parser_type="PydanticOutputParser",
            )

        json_str = self._extract_json(text)

        if not json_str:
            raise OutputParserException(
                f"无法从文本中提取 JSON。文本前 150 字符: {text[:150]}",
                text=text,
                parser_type="PydanticOutputParser",
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise OutputParserException(
                f"JSON 解析失败: {e}。提取到的 JSON 片段: {json_str[:200]}",
                text=text,
                parser_type="PydanticOutputParser",
            )

        if not isinstance(data, dict):
            raise OutputParserException(
                f"期望 JSON 对象，实际得到 {type(data).__name__}: {str(data)[:100]}",
                text=text,
                parser_type="PydanticOutputParser",
            )

        try:
            return self.pydantic_object(**data)
        except Exception as e:
            raise OutputParserException(
                f"Pydantic 模型 '{self.pydantic_object.__name__}' 验证失败: {e}\n"
                f"解析到的数据: {data}",
                text=text,
                parser_type="PydanticOutputParser",
            )

    def get_format_instructions(self) -> str:
        """从 Pydantic 模型字段生成 JSON Schema 格式说明。

        输出包含字段名、类型、描述、是否必填等信息，
        可直接追加到 System Prompt 中使用。
        """
        schema = self._get_schema()
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        model_name = self.pydantic_object.__name__
        lines = [
            f"请严格按以下 JSON 格式输出（{model_name}），不要添加其他文字:",
            "{",
        ]

        for field_name, field_info in props.items():
            field_type = field_info.get("type", "string")
            # 处理数组类型
            if field_type == "array":
                items = field_info.get("items", {})
                item_type = items.get("type", "string")
                field_type = f"{item_type}[]"

            field_desc = field_info.get("description", "")
            desc_part = f"  // {field_desc}" if field_desc else ""
            req_mark = " 【必填】" if field_name in required else " 【可选】"

            lines.append(f'  "{field_name}": <{field_type}>{req_mark}{desc_part}')

        lines.append("}")
        return "\n".join(lines)

    # ---- 私有 ----

    def _validate_pydantic(self):
        """验证 pydantic_object 是有效的 Pydantic BaseModel 子类。"""
        try:
            from pydantic import BaseModel

            if not issubclass(self.pydantic_object, BaseModel):
                raise OutputParserException(
                    f"pydantic_object 必须是 Pydantic BaseModel 的子类，"
                    f"实际类型: {self.pydantic_object}",
                    parser_type="PydanticOutputParser",
                )
        except ImportError:
            raise ImportError(
                "使用 PydanticOutputParser 需要安装 pydantic。\n"
                "请运行: pip install pydantic"
            )

    def _get_schema(self) -> dict:
        """获取 Pydantic 模型的 JSON Schema（兼容 v1/v2）。"""
        try:
            # Pydantic v2
            return self.pydantic_object.model_json_schema()
        except AttributeError:
            # Pydantic v1
            return self.pydantic_object.schema()

    @staticmethod
    def _extract_json(text: str) -> str:
        """从文本中提取第一个完整的 JSON 对象字符串。

        支持:
        - Markdown 代码块: ```json ... ```
        - 嵌入式 JSON 对象
        - 处理嵌套大括号和字符串内的特殊字符
        """
        # 先尝试从 Markdown 代码块中提取
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_match:
            text = code_match.group(1).strip()

        start = text.find("{")
        if start == -1:
            return ""

        # 用状态机精确匹配括号（处理字符串内的 {} ）
        in_string = False
        escape = False
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

        return text[start:]

    def __repr__(self) -> str:
        model_name = getattr(self.pydantic_object, "__name__", str(self.pydantic_object))
        return f"PydanticOutputParser(pydantic_object={model_name})"
