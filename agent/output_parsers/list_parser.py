"""
CommaSeparatedListOutputParser —— 列表输出解析器。

将 LLM 输出的列表文本解析为 Python list[str]。
支持多种常见的列表表示格式。
"""

import json
import re
from typing import Any

from agent.output_parsers.base import BaseOutputParser, OutputParserException


class CommaSeparatedListOutputParser(BaseOutputParser):
    """逗号分隔列表解析器。

    将 LLM 输出的列表文本解析为 Python list[str]。
    自动识别并处理以下格式:
    - 逗号分隔: "上海, 北京, 广州"
    - 中文逗号: "上海，北京，广州"
    - 编号列表: "1. 上海\\n2. 北京\\n3. 广州"
    - 带引号: '"上海", "北京", "广州"'
    - JSON 数组: '["上海", "北京", "广州"]'
    - 换行分隔: 每行一个条目

    Parameters:
        separator: 主分隔符，默认逗号 ","
        strip: 是否去除每个元素前后的空白和引号，默认 True
        drop_empty: 是否丢弃空字符串元素，默认 True

    使用示例:
        parser = CommaSeparatedListOutputParser()
        result = parser.parse("上海, 北京, 广州")
        print(result)  # ["上海", "北京", "广州"]

        result = parser.parse("1. 天气查询\\n2. 翻译\\n3. 计算")
        print(result)  # ["天气查询", "翻译", "计算"]
    """

    def __init__(self, separator: str = ",", strip: bool = True, drop_empty: bool = True):
        super().__init__()
        self.separator = separator
        self.strip = strip
        self.drop_empty = drop_empty

    def parse(self, text: str) -> list[str]:
        """将文本解析为字符串列表。

        Raises:
            OutputParserException: 输入为空且不跳过空输入时
        """
        text = text.strip()

        if not text:
            if self.skip_empty:
                return []
            raise OutputParserException(
                "输入文本为空，无法解析为列表。",
                text=text,
                parser_type="CommaSeparatedListOutputParser",
            )

        # 尝试 JSON 数组格式: ["a", "b", "c"]
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item or not self.drop_empty]
            except (json.JSONDecodeError, ValueError):
                pass

        # 尝试编号列表: "1. item1\n2. item2" 或 "1) item1\n2) item2"
        numbered_match = re.findall(r'^\d+[\.\)、]\s*(.+?)$', text, re.MULTILINE)
        if len(numbered_match) >= 2:
            items = numbered_match
        else:
            # 尝试中文逗号分隔
            if "，" in text and self.separator == ",":
                items = text.split("，")
            else:
                items = text.split(self.separator)

        if self.strip:
            items = [
                item.strip().strip('"').strip("'").strip("「").strip("」")
                for item in items
            ]
        else:
            items = [item.strip('"').strip("'") for item in items]

        if self.drop_empty:
            items = [item for item in items if item]

        return items

    def get_format_instructions(self) -> str:
        """生成列表格式说明。"""
        return (
            f"请以{self.separator}分隔的列表格式输出，每项为简短名称，不要编号或其他文字。\n"
            f"示例: 项目A{self.separator} 项目B{self.separator} 项目C"
        )

    def __repr__(self) -> str:
        return f"CommaSeparatedListOutputParser(separator='{self.separator}')"
