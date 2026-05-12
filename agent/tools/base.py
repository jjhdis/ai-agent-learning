"""
Tool 基类 —— 所有工具只需继承此基类并实现 execute 方法即可。

新增一个 Tool 的标准步骤:
    1. 继承 BaseTool
    2. 设置 name / description / parameters
    3. 实现 execute(**kwargs) -> str
    4. 在 registry.py 中注册（或在初始化 Agent 时手动注册）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """单个参数的描述。"""
    name: str
    type: str          # "string" | "number" | "integer" | "boolean"
    description: str
    required: bool = True
    enum: list[str] = None  # 可选，限定参数取值范围


class BaseTool(ABC):
    """
    Tool 基类。

    示例 —— 实现一个计算器 Tool:
        class CalculatorTool(BaseTool):
            name = "calculator"
            description = "执行四则运算。输入一个数学表达式字符串。"
            parameters = [
                ToolParameter("expression", "string", "数学表达式，如 '2 + 3 * 4'"),
            ]

            def execute(self, expression: str) -> str:
                return str(eval(expression))
    """

    name: str = ""
    description: str = ""
    parameters: list[ToolParameter] = []

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具逻辑，返回字符串结果。"""
        ...

    def to_openai_function(self) -> dict:
        """转换为 OpenAI function calling 格式。"""
        properties = {}
        required: list[str] = []

        for p in self.parameters:
            prop: dict = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
