"""
StreamingCallbackHandler —— 流式回调处理器。

在流式输出过程中实时打印/记录各类事件。
可组合到 Agent 的 callbacks 中，与现有 LoggingCallback 等并存。
"""

import time

from agent.callback.base import BaseCallbackHandler
from agent.streaming.event import StreamEventType


class StreamingCallbackHandler(BaseCallbackHandler):
    """流式回调处理器 —— 监听流式会话中的实时事件。

    与现有的 LoggingCallback 并行工作，不修改原有回调行为。
    支持三种输出模式:
    - realtime: 实时逐字打印（适合终端交互）
    - silent:   静默收集（适合后台处理）
    - verbose:  详细模式（打印所有事件，适合调试）

    Parameters:
        mode: 输出模式 "realtime" / "silent" / "verbose"
        show_thinking: realtime 模式下是否显示思考过程，默认 False

    使用示例:
        handler = StreamingCallbackHandler(mode="realtime")
        agent = Agent(..., callbacks=CallbackManager([handler]))
        for event in agent.run_stream("上海天气"):
            handler.handle_event(event)  # 实时打印
    """

    def __init__(self, mode: str = "realtime", show_thinking: bool = False):
        self.mode = mode
        self.show_thinking = show_thinking
        self._events: list = []
        self._start_time = None
        self._token_count = 0
        self._tool_count = 0
        self._llm_call_count = 0

    # ---- 公共 ----

    def handle_event(self, event) -> None:
        """处理一个 StreamEvent，根据 mode 执行相应动作。

        可随 Agent.run_stream() 的迭代同步调用，
        也可通过回调系统自动触发。
        """
        self._events.append(event)

        if event.event == StreamEventType.THINK:
            self._handle_think(event)
        elif event.event == StreamEventType.REPLY:
            self._handle_reply(event)
        elif event.event == StreamEventType.TOOL_START:
            self._handle_tool_start(event)
        elif event.event == StreamEventType.TOOL_END:
            self._handle_tool_end(event)
        elif event.event == StreamEventType.TOOL_ERROR:
            self._handle_tool_error(event)
        elif event.event == StreamEventType.DONE:
            self._handle_done(event)
        elif event.event == StreamEventType.ERROR:
            self._handle_error(event)

    def reset(self) -> None:
        """重置统计计数器。"""
        self._events.clear()
        self._start_time = None
        self._token_count = 0
        self._tool_count = 0
        self._llm_call_count = 0

    # ---- 统计属性 ----

    @property
    def token_count(self) -> int:
        """已接收的文本 token 总数。"""
        return self._token_count

    @property
    def tool_count(self) -> int:
        """已执行的工具调用次数。"""
        return self._tool_count

    @property
    def llm_call_count(self) -> int:
        """流式 LLM 调用次数（每轮 ReAct 推理计一次）。"""
        return self._llm_call_count

    @property
    def elapsed(self) -> float:
        """从第一个事件到现在的耗时（秒）。"""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    # ---- 内部 ----

    def _handle_think(self, event) -> None:
        if self._start_time is None:
            self._start_time = time.time()
        self._token_count += len(str(event.data or ""))
        if self.mode == "verbose":
            print(f"[思考] {event.data}", end="", flush=True)
        elif self.mode == "realtime" and self.show_thinking:
            print(f"\033[90m{event.data}\033[0m", end="", flush=True)

    def _handle_reply(self, event) -> None:
        if self._start_time is None:
            self._start_time = time.time()
        self._token_count += len(str(event.data or ""))
        if self.mode in ("realtime", "verbose"):
            print(event.data, end="", flush=True)

    def _handle_tool_start(self, event) -> None:
        self._tool_count += 1
        if self.mode == "verbose":
            data = event.data or {}
            print(f"\n[工具调用] {data.get('name', '?')}({data.get('args', {})})")

    def _handle_tool_end(self, event) -> None:
        if self.mode == "verbose":
            data = event.data or {}
            result_preview = str(data.get("result", ""))[:100]
            print(f"[工具结果] {data.get('name', '?')}: {result_preview}")

    def _handle_tool_error(self, event) -> None:
        if self.mode in ("realtime", "verbose"):
            data = event.data or {}
            print(f"\n[工具错误] {data.get('name', '?')}: {data.get('error', '')}")

    def _handle_done(self, event) -> None:
        if self.mode in ("realtime", "verbose"):
            print()  # 换行

    def _handle_error(self, event) -> None:
        if self.mode in ("realtime", "verbose"):
            print(f"\n[流式错误] {event.data}")

    def __repr__(self) -> str:
        return (
            f"StreamingCallbackHandler(mode={self.mode}, "
            f"tokens={self._token_count}, tools={self._tool_count})"
        )
