"""
CallbackManager —— 回调管理器。

管理一组回调处理器，在生命周期关键节点将事件分发给所有已注册的处理器。
与 LangChain 的 CallbackManager 设计一致。
"""

from agent.callback.base import BaseCallbackHandler


class CallbackManager:
    """管理多个回调处理器，支持运行时添加/移除。

    所有 on_xxx 方法都会遍历已注册的处理器列表并依次调用，
    某个处理器的异常不会中断其他处理器的执行。

    使用示例:
        manager = CallbackManager()
        manager.add_handler(LoggingCallback())
        manager.add_handler(MyCustomCallback())
    """

    def __init__(self, handlers: list[BaseCallbackHandler] = None):
        self.handlers: list[BaseCallbackHandler] = list(handlers) if handlers else []

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """注册一个回调处理器。"""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """移除一个回调处理器。"""
        if handler in self.handlers:
            self.handlers.remove(handler)

    # ---- 内部 ----

    def _dispatch(self, event: str, *args, **kwargs) -> None:
        """向所有处理器分发事件，单个处理器的异常不中断其他处理器。"""
        for handler in self.handlers:
            try:
                method = getattr(handler, event, None)
                if method:
                    method(*args, **kwargs)
            except Exception:
                pass

    # ---- Agent 级事件 ----

    def on_agent_start(self, message: str) -> None:
        self._dispatch("on_agent_start", message)

    def on_agent_end(self, reply: str) -> None:
        self._dispatch("on_agent_end", reply)

    def on_agent_error(self, error: Exception) -> None:
        self._dispatch("on_agent_error", error)

    # ---- LLM 级事件 ----

    def on_llm_start(self, messages: list[dict], tools: list[dict] = None) -> None:
        self._dispatch("on_llm_start", messages, tools)

    def on_llm_end(self, response) -> None:
        self._dispatch("on_llm_end", response)

    def on_llm_error(self, error: Exception) -> None:
        self._dispatch("on_llm_error", error)

    def on_think(self, content: str) -> None:
        self._dispatch("on_think", content)

    # ---- Tool 级事件 ----

    def on_tool_start(self, name: str, args: dict) -> None:
        self._dispatch("on_tool_start", name, args)

    def on_tool_end(self, name: str, result: str) -> None:
        self._dispatch("on_tool_end", name, result)

    def on_tool_error(self, name: str, error: Exception) -> None:
        self._dispatch("on_tool_error", name, error)
