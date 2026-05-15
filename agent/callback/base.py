"""
BaseCallbackHandler —— 回调处理器基类。

所有回调处理器只需继承此基类并覆写需要监听的事件方法。
未覆写的方法默认不做任何操作（no-op）。
"""


class BaseCallbackHandler:
    """回调处理器基类，定义所有可监听的生命周期钩子。

    与 LangChain 的 BaseCallbackHandler 设计一致：
    子类只需覆写自己关心的事件方法，其余默认为空操作。

    事件分类：
        Agent 级：on_agent_start / on_agent_end / on_agent_error
        LLM 级：  on_llm_start / on_llm_end / on_llm_error / on_think
        Tool 级： on_tool_start / on_tool_end / on_tool_error
    """

    # ---- Agent 级事件 ----

    def on_agent_start(self, message: str) -> None:
        """Agent 开始处理用户消息时触发。"""

    def on_agent_end(self, reply: str) -> None:
        """Agent 返回最终回复时触发。"""

    def on_agent_error(self, error: Exception) -> None:
        """Agent 执行过程中出错时触发。"""

    # ---- LLM 级事件 ----

    def on_llm_start(self, messages: list[dict], tools: list[dict] = None) -> None:
        """即将调用 LLM 时触发，可获取当前消息历史和工具定义。"""

    def on_llm_end(self, response) -> None:
        """LLM 调用完成时触发，可获取 OpenAI ChatCompletionMessage。"""

    def on_llm_error(self, error: Exception) -> None:
        """LLM 调用出错时触发。"""

    def on_think(self, content: str) -> None:
        """LLM 在推理过程中输出文本内容时触发（非最终回复）。"""

    # ---- Tool 级事件 ----

    def on_tool_start(self, name: str, args: dict) -> None:
        """即将执行工具时触发。"""

    def on_tool_end(self, name: str, result: str) -> None:
        """工具执行完成时触发。"""

    def on_tool_error(self, name: str, error: Exception) -> None:
        """工具执行出错时触发。"""
