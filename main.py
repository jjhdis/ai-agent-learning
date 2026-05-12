"""
AI Agent 入口 —— 交互式对话。

使用前设置环境变量（秘钥你来填）：
    export LLM_API_KEY="sk-your-deepseek-key"

然后运行:
    python main.py
"""

import sys

from config import Config
from agent.llm.client import LLMClient
from agent.tools.registry import ToolRegistry
from agent.tools.weather import WeatherTool
from agent.core.agent import Agent


def build_agent() -> Agent:
    """组装 Agent：配置 LLM + 注册工具。"""
    llm = LLMClient(Config.llm)
    registry = ToolRegistry()
    registry.register(WeatherTool())

    return Agent(
        llm_client=llm,
        registry=registry,
        hooks={
            "on_start": lambda msg: print(f"\n{'='*50}\n[You]: {msg}"),
            "on_tool_call": lambda name, args: print(f"[Tool] 调用工具: {name}({args})"),
            "on_reply": lambda msg: print(f"[AI]: {msg}"),
        },
    )


def main():
    print("=" * 50)
    print(f"[AI Agent 已启动]")
    print(f"   模型: {Config.llm.model}")
    print(f"   工具: 天气查询 (get_weather)")
    print(f"   输入 '/quit' 退出, '/reset' 清空历史")
    print("=" * 50)

    agent = build_agent()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[再见!]")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("[再见!]")
            break
        if user_input.lower() == "/reset":
            agent.reset()
            print("[对话历史已清空]")
            continue

        try:
            agent.run(user_input)
        except Exception as e:
            print(f"[错误] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
