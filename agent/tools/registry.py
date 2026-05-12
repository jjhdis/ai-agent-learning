"""
Tool 注册中心 —— 管理所有可用工具。
"""

from agent.tools.base import BaseTool


class ToolRegistry:
    """工具注册中心。

    新增工具后，调用 registry.register(MyTool()) 即可接入 Agent。
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具。同名工具会被覆盖。"""
        self._tools[tool.name] = tool

    def get(self, name: str):
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_openai_definitions(self) -> list[dict]:
        """返回所有工具的 OpenAI function calling 格式定义。"""
        return [t.to_openai_function() for t in self._tools.values()]
