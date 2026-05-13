"""
Agent 核心引擎 —— ReAct（Reasoning + Acting）循环。

流程:
    User 输入 → LLM 推理 → [需调工具?] → 执行工具 → 回传结果 → LLM 再次推理
                                                               ↓
                                                          直接回复 User

扩展方式:
    - 新增工具: 实现 BaseTool → 注册到 ToolRegistry → Agent 自动感知
    - 替换模型: 修改 Config.llm，无需改动 Agent 代码
    - 交互模式: 继承 Agent 重写 on_start / on_think / on_tool_call / on_reply 钩子
"""

import json
from typing import Callable

from agent.chain.runnable import Runnable
from agent.llm.client import LLMClient
from agent.tools.registry import ToolRegistry


class Agent(Runnable):
    """可扩展的 AI Agent，实现 Runnable 协议。

    Parameters:
        llm_client: LLM 客户端
        registry: 工具注册中心
        max_iterations: 单次请求最多调用工具的次数（防止无限循环）
        hooks: 可选的生命周期回调
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ToolRegistry,
        max_iterations: int = 5,
        hooks: dict[str, Callable] = None,
    ):
        self.llm = llm_client
        self.registry = registry
        self.max_iterations = max_iterations
        self.hooks = hooks or {}
        self.history: list[dict] = []

    # ---- 公共 ----

    def run(self, user_message: str) -> str:
        """处理一条用户消息，返回 Agent 的最终文本回复。"""
        print(f"\n{'='*60}")
        print(f"[Agent] 收到用户消息: {user_message}")
        print(f"{'='*60}\n")
        self._emit("on_start", user_message)

        self.history.append({"role": "user", "content": user_message})
        tools = self.registry.get_openai_definitions()

        if tools:
            print(f"[Agent] 已注册工具: {[t['function']['name'] for t in tools]}")
        else:
            print("[Agent] 未注册任何工具")

        for step in range(self.max_iterations):
            print(f"\n{'─'*50}")
            print(f"[Agent] 第 {step + 1} 轮推理 (共 {self.max_iterations} 轮)")
            print(f"{'─'*50}")
            print(f"[Agent] >>> 发送给 LLM 的历史消息数: {len(self.history)}")
            print(f"[Agent] >>> 历史消息概览:")
            for i, h in enumerate(self.history):
                role = h["role"]
                content_preview = (h.get("content", "") or "")[:80]
                print(f"         [{i}] {role}: {content_preview}")
            print(f"[Agent] >>> 可用工具定义: {[t['function']['name'] for t in tools] if tools else '无'}")

            msg = self.llm.chat(self.history, tools if tools else None)

            print(f"\n[Agent] <<< LLM 返回:")
            if msg.content:
                print(f"         content: {msg.content[:200]}")
            else:
                print(f"         content: (无)")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"         tool_call: {tc.function.name}({tc.function.arguments})")
            else:
                print(f"         tool_calls: (无)")

            if msg.content:
                self._emit("on_think", msg.content)

            if not msg.tool_calls:
                # LLM 给出最终文本回复
                print(f"\n[Agent] ✓ LLM 给出最终回复，结束循环")
                self.history.append({"role": "assistant", "content": msg.content})
                self._emit("on_reply", msg.content)
                print(f"[Agent] 最终回复: {msg.content[:200]}")
                print(f"{'='*60}\n")
                return msg.content

            # LLM 要求调用工具
            print(f"\n[Agent] → LLM 要求调用工具，准备执行...")
            self.history.append(msg.model_dump())
            self._handle_tool_calls(msg.tool_calls)

        print(f"\n[Agent] ✗ 已达到最大推理步数 ({self.max_iterations})，强制结束")
        return "已达到最大推理步数，请简化问题后重试。"

    def invoke(self, user_message: str) -> str:
        """Runnable 协议接口，与 run() 等价。"""
        return self.run(user_message)

    def reset(self) -> None:
        """清空对话历史。"""
        self.history.clear()

    # ---- 私有 ----

    def _handle_tool_calls(self, tool_calls) -> None:
        """执行工具调用并将结果追加到历史。"""
        results = []
        for tc in tool_calls:
            tool = self.registry.get(tc.function.name)
            if not tool:
                content = f"[错误] 未知工具: {tc.function.name}"
            else:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    content = f"[错误] 参数解析失败: {tc.function.arguments}"
                else:
                    self._emit("on_tool_call", tool.name, args)
                    content = tool.execute(**args)

            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

        self.history.extend(results)

    def _emit(self, event: str, *args) -> None:
        """触发钩子回调。"""
        if fn := self.hooks.get(event):
            fn(*args)
