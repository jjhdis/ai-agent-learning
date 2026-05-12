"""
LLM 客户端 —— 封装 OpenAI 兼容接口（DeepSeek / Qwen / GLM / OpenAI 等）。
"""

from openai import OpenAI

from config import LLMConfig


class LLMClient:
    """统一的 LLM 调用入口，支持所有 OpenAI 兼容服务。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def chat(self, messages: list[dict], tools: list[dict] = None):
        """
        发送对话请求，返回 OpenAI ChatCompletionMessage。

        调用方通过返回消息中的 tool_calls / content 判读下一步行为。
        """
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message
