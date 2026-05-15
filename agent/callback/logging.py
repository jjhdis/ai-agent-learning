"""
LoggingCallback —— 日志与追踪回调处理器。

记录每次 Agent 运行的耗时、LLM 调用次数、工具调用次数等指标，
可用于性能监控和调试。
"""

import time

from agent.callback.base import BaseCallbackHandler


class LoggingCallback(BaseCallbackHandler):
    """记录 Agent 运行日志和性能统计。

    统计指标：
        - 总耗时（elapsed）
        - LLM 调用次数（llm_call_count）
        - 工具调用次数（tool_call_count）

    Parameters:
        verbose: 是否打印详细日志，默认 True。
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._start_time: float = 0
        self._llm_call_count: int = 0
        self._tool_call_count: int = 0

    # ---- 只读统计属性 ----

    @property
    def llm_call_count(self) -> int:
        return self._llm_call_count

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    @property
    def elapsed(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def reset(self) -> None:
        """重置所有统计计数器。"""
        self._start_time = 0
        self._llm_call_count = 0
        self._tool_call_count = 0

    # ---- Agent 级事件 ----

    def on_agent_start(self, message: str) -> None:
        self.reset()
        self._start_time = time.time()
        if self.verbose:
            print(f"[Callback] Agent 启动 — 用户消息: {message}")

    def on_agent_end(self, reply: str) -> None:
        if self.verbose:
            preview = reply[:80].replace("\n", " ")
            print(
                f"[Callback] Agent 完成 — "
                f"耗时 {self.elapsed:.2f}s, "
                f"LLM 调用 {self._llm_call_count} 次, "
                f"工具调用 {self._tool_call_count} 次, "
                f"回复预览: {preview}..."
            )

    def on_agent_error(self, error: Exception) -> None:
        if self.verbose:
            print(f"[Callback] Agent 出错 — {type(error).__name__}: {error}")

    # ---- LLM 级事件 ----

    def on_llm_start(self, messages: list[dict], tools: list[dict] = None) -> None:
        self._llm_call_count += 1
        if self.verbose:
            tool_names = [t["function"]["name"] for t in tools] if tools else []
            print(
                f"[Callback] LLM #{self._llm_call_count} 开始 — "
                f"消息数: {len(messages)}, 可用工具: {tool_names}"
            )

    def on_llm_end(self, response) -> None:
        if self.verbose:
            content_len = len(response.content) if response.content else 0
            tool_calls_count = len(response.tool_calls) if response.tool_calls else 0
            print(
                f"[Callback] LLM #{self._llm_call_count} 返回 — "
                f"content 长度: {content_len}, tool_calls 数: {tool_calls_count}"
            )

    def on_llm_error(self, error: Exception) -> None:
        if self.verbose:
            print(f"[Callback] LLM #{self._llm_call_count} 出错 — {type(error).__name__}: {error}")

    def on_think(self, content: str) -> None:
        if self.verbose:
            preview = content[:100].replace("\n", " ")
            print(f"[Callback] LLM 思考 — {preview}...")

    # ---- Tool 级事件 ----

    def on_tool_start(self, name: str, args: dict) -> None:
        self._tool_call_count += 1
        if self.verbose:
            print(f"[Callback] 工具 #{self._tool_call_count} 开始 — {name}({args})")

    def on_tool_end(self, name: str, result: str) -> None:
        if self.verbose:
            preview = result[:80].replace("\n", " ")
            print(f"[Callback] 工具 #{self._tool_call_count} 完成 — {name}: {preview}...")

    def on_tool_error(self, name: str, error: Exception) -> None:
        if self.verbose:
            print(f"[Callback] 工具 #{self._tool_call_count} 出错 — {name}: {type(error).__name__}: {error}")
