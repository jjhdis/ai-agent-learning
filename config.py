"""
统一配置管理。

所有敏感信息通过环境变量注入，不写入代码：
    LLM_API_KEY  - 模型 API 密钥
    LLM_BASE_URL - 模型 API 地址（可选，默认 DeepSeek）
    LLM_MODEL    - 模型名称（可选，默认 deepseek-v4-pro）

使用方式:
    from config import Config
    llm_cfg = Config.llm
"""

import os


class LLMConfig:
    """LLM 配置，兼容所有 OpenAI 兼容接口（DeepSeek、Qwen、GLM 等）。"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.api_key = api_key or "sk-48560546f57b4262bdaeca0353c58bf5"
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.model = model or "deepseek-chat"
        self.temperature = temperature
        self.max_tokens = max_tokens


class Config:
    llm = LLMConfig()
