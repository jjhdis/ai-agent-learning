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
        发送对话请求，返回 OpenAI ChatCompletion 完整响应。

        返回完整响应对象（包含 usage 信息），调用方可通过 .choices[0].message
        获取消息内容，通过 .usage 获取 Token 消耗统计。

        兼容旧代码：msg.content / msg.tool_calls / msg.model_dump() 仍可正常使用。
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
        return response
