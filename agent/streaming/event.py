"""
StreamEvent —— 流式事件数据模型与流式块累加器。

定义 Agent.run_stream() 产出的所有事件类型，以及
StreamAccumulator 用于将 OpenAI 流式 delta 块重组为完整消息。
"""

from dataclasses import dataclass, field
from typing import Any


class StreamEventType:
    """流式事件类型常量。

    think:       LLM 推理过程中的文本 token（非最终回复）
    reply:       最终回复的文本 token
    tool_start:  工具即将开始执行
    tool_end:    工具执行完成
    tool_error:  工具执行出错
    done:        流式输出全部完成
    error:       流式过程中发生错误
    """

    THINK = "think"
    REPLY = "reply"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamEvent:
    """流式事件，由 Agent.run_stream() 产出。

    Attributes:
        event: 事件类型，取值见 StreamEventType
        data: 事件携带的数据，根据事件类型不同:
              - think/reply: str (文本 token)
              - tool_start: dict {"name": str, "args": dict}
              - tool_end: dict {"name": str, "result": str}
              - tool_error: dict {"name": str, "error": str}
              - done: str (完整回复文本)
              - error: str (错误信息)
        step: 当前 ReAct 循环步数（从 0 开始）
        metadata: 额外元数据

    使用示例:
        for event in agent.run_stream("上海天气"):
            if event.event == StreamEventType.REPLY:
                print(event.data, end="", flush=True)  # 逐字输出
            elif event.event == StreamEventType.THINK:
                print(f"[思考] {event.data}", end="")
    """

    event: str
    data: Any = None
    step: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        data_preview = str(self.data)[:60] if self.data else "None"
        return f"StreamEvent({self.event}, step={self.step}, data={data_preview})"


class StreamAccumulator:
    """流式块累加器 —— 将 OpenAI 流式 delta 块重组为完整消息。

    OpenAI 流式 API 的 tool_calls 跨多个 chunk 分片到达:
        chunk_1: delta.tool_calls = [{"index": 0, "id": "call_xxx",
                                       "function": {"name": "get_", "arguments": ""}}]
        chunk_2: delta.tool_calls = [{"index": 0,
                                       "function": {"arguments": "weather"}}]
        ...

    本累加器按 index 合并这些分片，生成完整的 assistant 消息 dict。

    使用示例:
        acc = StreamAccumulator()
        for chunk in llm.stream_chat(messages, tools):
            acc.add_chunk(chunk)
            for token in acc.new_tokens():
                print(token, end="", flush=True)

        if acc.has_tool_calls:
            msg = acc.build_message()  # → {"role": "assistant", "tool_calls": [...]}
        else:
            reply = acc.content        # → str 完整回复文本
    """

    def __init__(self):
        self.content = ""
        self._content_index = 0  # 追踪已返回的 token 位置
        # tool_calls 累积: {index: {id, type, function: {name, arguments}}}
        self._tool_calls: dict[int, dict] = {}
        self.finish_reason = None
        self.usage = None
        self._done = False

    # ---- 属性 ----

    @property
    def has_tool_calls(self) -> bool:
        """流式响应是否包含工具调用。"""
        return len(self._tool_calls) > 0

    @property
    def tool_calls(self) -> list[dict]:
        """获取累积完成的工具调用列表（OpenAI 格式）。"""
        result = []
        for idx in sorted(self._tool_calls):
            tc = self._tool_calls[idx]
            result.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            })
        return result

    @property
    def is_done(self) -> bool:
        """流式响应是否已完成。"""
        return self._done

    # ---- 核心方法 ----

    def add_chunk(self, chunk) -> None:
        """处理一个 OpenAI 流式 chunk。

        Args:
            chunk: OpenAI ChatCompletionChunk 对象
        """
        if self._done:
            return

        # 处理 usage（仅在 stream_options={"include_usage": True} 时出现）
        if hasattr(chunk, "usage") and chunk.usage:
            self.usage = chunk.usage

        choices = getattr(chunk, "choices", None)
        if not choices:
            return

        delta = choices[0].delta if choices else None
        if not delta:
            return

        # 累积文本内容
        if hasattr(delta, "content") and delta.content:
            self.content += delta.content

        # 累积工具调用
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in self._tool_calls:
                    self._tool_calls[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                acc = self._tool_calls[idx]
                if tc_delta.id:
                    acc["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        acc["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        acc["function"]["arguments"] += tc_delta.function.arguments

        # 检查是否结束
        finish = getattr(choices[0], "finish_reason", None)
        if finish:
            self.finish_reason = finish
            self._done = True

    def new_tokens(self) -> str:
        """返回自上次调用以来新增的文本 token。

        每次调用后内部指针前移，适合在消费循环中逐次获取增量文本。
        """
        new_text = self.content[self._content_index:]
        self._content_index = len(self.content)
        return new_text

    def build_message(self) -> dict:
        """构建完整的 assistant 消息 dict。

        Returns:
            OpenAI 格式的消息字典，包含 tool_calls（如有）或 content

        Raises:
            RuntimeError: 流式响应尚未完成时调用
        """
        if not self._done:
            raise RuntimeError("流式响应尚未完成，无法构建消息。")

        msg = {"role": "assistant"}

        if self.has_tool_calls:
            msg["tool_calls"] = self.tool_calls
            if self.content:
                msg["content"] = self.content
        else:
            msg["content"] = self.content

        return msg

    def reset(self) -> None:
        """重置累加器状态，准备处理下一次流式请求。"""
        self.content = ""
        self._content_index = 0
        self._tool_calls.clear()
        self.finish_reason = None
        self.usage = None
        self._done = False

    def __repr__(self) -> str:
        status = "done" if self._done else "accumulating"
        tc_info = f", tool_calls={len(self._tool_calls)}" if self._tool_calls else ""
        return f"StreamAccumulator({status}, content_len={len(self.content)}{tc_info})"
