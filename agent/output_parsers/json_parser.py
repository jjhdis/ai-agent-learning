"""
JsonOutputParser —— JSON 输出解析器。

从 LLM 的文本输出中提取 JSON 对象并解析为 Python dict。
支持多种常见输出格式，并可通过 expected_keys 生成格式引导说明。
"""

import json
import re
from typing import Any

from agent.output_parsers.base import BaseOutputParser, OutputParserException


class JsonOutputParser(BaseOutputParser):
    """JSON 输出解析器。

    从 LLM 的文本输出中智能提取并解析 JSON 对象。
    自动处理以下情况:
    - 纯 JSON 文本: '{"key": "value"}'
    - Markdown 代码块中的 JSON: ```json\\n{...}\\n```
    - 嵌在普通文本中的 JSON 对象
    - JSON 数组: '["a", "b"]' → 包装为 {"result": [...]}

    Parameters:
        expected_keys: 期望的 JSON 键名列表，用于生成格式说明
        array_as_result: JSON 数组是否包装为 {"result": [...]}，默认 True

    使用示例:
        parser = JsonOutputParser(expected_keys=["city", "temperature", "condition"])
        instructions = parser.get_format_instructions()
        # 将 instructions 追加到 System Prompt 引导 LLM 输出...
        result = parser.parse('{"city": "上海", "temperature": 28.0, "condition": "晴"}')
        print(result["city"])  # "上海"
    """

    def __init__(self, expected_keys: list[str] = None, array_as_result: bool = True):
        super().__init__()
        self.expected_keys = expected_keys or []
        self.array_as_result = array_as_result

    def parse(self, text: str) -> dict:
        """从文本中提取并解析 JSON。

        Raises:
            OutputParserException: 无法提取有效的 JSON
        """
        text = text.strip()
        if not text:
            raise OutputParserException(
                "输入文本为空，无法解析 JSON。",
                text=text,
                parser_type="JsonOutputParser",
            )

        # 尝试直接解析
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and self.array_as_result:
                return {"result": parsed}
            # 其他类型，包装返回
            return {"result": parsed}
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试从 Markdown 代码块中提取
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_match:
            try:
                parsed = json.loads(code_match.group(1).strip())
                return parsed if isinstance(parsed, dict) else {"result": parsed}
            except (json.JSONDecodeError, ValueError):
                pass

        # 尝试提取 JSON 对象（处理嵌套大括号）
        json_str = self._extract_balanced_json(text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                return parsed if isinstance(parsed, dict) else {"result": parsed}
            except (json.JSONDecodeError, ValueError):
                pass

        # 尝试修复常见错误：单引号替换为双引号
        fixed = self._try_fix_json(text)
        if fixed:
            try:
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                pass

        raise OutputParserException(
            f"无法从文本中提取有效的 JSON。文本前 150 字符: {text[:150]}",
            text=text,
            parser_type="JsonOutputParser",
        )

    def get_format_instructions(self) -> str:
        """生成 JSON 格式说明，可追加到 System Prompt。"""
        if self.expected_keys:
            fields_desc = "\n".join(
                f'    "{key}": <{key}的值>' for key in self.expected_keys
            )
            return (
                "请严格按以下 JSON 格式输出，不要添加 Markdown 代码块标记或其他文字:\n"
                "{\n" + fields_desc + "\n}"
            )
        return (
            "请以 JSON 格式输出，不要添加其他文字。\n"
            '示例: {"key1": "值1", "key2": "值2"}'
        )

    # ---- 私有 ----

    @staticmethod
    def _extract_balanced_json(text: str) -> str:
        """用括号计数提取第一个完整的 JSON 对象或数组。"""
        # 找第一个 { 或 [
        for start_char in ("{", "["):
            start = text.find(start_char)
            if start == -1:
                continue
            end_char = "}" if start_char == "{" else "]"
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
                    if c == start_char:
                        depth += 1
                    elif c == end_char:
                        depth -= 1
                        if depth == 0:
                            return text[start : i + 1]
        return ""

    @staticmethod
    def _try_fix_json(text: str) -> str:
        """尝试修复常见的 JSON 格式错误。"""
        # 提取花括号内容
        start = text.find("{")
        if start == -1:
            return ""
        end = text.rfind("}")
        if end == -1:
            return ""
        candidate = text[start : end + 1]
        # 将 Python 风格的 None/True/False 替换为 JSON 风格
        candidate = re.sub(r"\bNone\b", "null", candidate)
        candidate = re.sub(r"\bTrue\b", "true", candidate)
        candidate = re.sub(r"\bFalse\b", "false", candidate)
        return candidate

    def __repr__(self) -> str:
        if self.expected_keys:
            return f"JsonOutputParser(expected_keys={self.expected_keys})"
        return "JsonOutputParser()"
