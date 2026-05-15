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
    - 跨轮记忆: 注入 BaseMemory 实例，Agent 自动在 run() 前后加载/保存历史
    - 回调监听: 注入 CallbackManager + 自定义 BaseCallbackHandler，监听所有生命周期事件
"""

import json
from typing import Callable

from agent.chain.runnable import Runnable
from agent.llm.client import LLMClient
from agent.tools.registry import ToolRegistry
from agent.memory.base import BaseMemory
from agent.callback.base import BaseCallbackHandler
from agent.callback.manager import CallbackManager


class _HooksAdapter(BaseCallbackHandler):
    """将旧版 hooks 字典适配为 BaseCallbackHandler，保证向后兼容。"""

    def __init__(self, hooks: dict[str, Callable]):
        self._hooks = hooks

    def on_agent_start(self, message: str) -> None:
        if fn := self._hooks.get("on_start"):
            fn(message)

    def on_think(self, content: str) -> None:
        if fn := self._hooks.get("on_think"):
            fn(content)

    def on_tool_start(self, name: str, args: dict) -> None:
        if fn := self._hooks.get("on_tool_call"):
            fn(name, args)

    def on_agent_end(self, reply: str) -> None:
        if fn := self._hooks.get("on_reply"):
            fn(reply)


class Agent(Runnable):
    """可扩展的 AI Agent，实现 Runnable 协议。

    Parameters:
        llm_client: LLM 客户端
        registry: 工具注册中心
        max_iterations: 单次请求最多调用工具的次数（防止无限循环）
        hooks: [已废弃] 旧版生命周期回调字典，请改用 callbacks 参数
        callbacks: CallbackManager 实例，管理一组回调处理器
        memory: 可选的对话记忆，为 None 时历史仅在单次 run() 内有效
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ToolRegistry,
        max_iterations: int = 5,
        hooks: dict[str, Callable] = None,
        callbacks: CallbackManager = None,
        memory: BaseMemory = None,
    ):
        self.llm = llm_client
        self.registry = registry
        self.max_iterations = max_iterations
        self.memory = memory
        self.history: list[dict] = []

        self.callbacks = callbacks or CallbackManager()
        if hooks:
            self.callbacks.add_handler(_HooksAdapter(hooks))

    # ---- 公共 ----

    def run(self, user_message: str) -> str:
        """处理一条用户消息，返回 Agent 的最终文本回复。"""
        print(f"\n{'='*60}")
        print(f"[Agent] 收到用户消息: {user_message}")
        print(f"{'='*60}\n")
        self.callbacks.on_agent_start(user_message)

        if self.memory:
            self.history = self.memory.load()
            print(f"[Agent] 从记忆加载 {len(self.history)} 条历史消息")

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

            self.callbacks.on_llm_start(self.history, tools if tools else None)

            try:
                response = self.llm.chat(self.history, tools if tools else None)
            except Exception as e:
                self.callbacks.on_llm_error(e)
                self.callbacks.on_agent_error(e)
                raise

            msg = response.choices[0].message
            self.callbacks.on_llm_end(response)

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
                self.callbacks.on_think(msg.content)

            if not msg.tool_calls:
                # LLM 给出最终文本回复
                print(f"\n[Agent] ✓ LLM 给出最终回复，结束循环")
                self.history.append({"role": "assistant", "content": msg.content})
                self.callbacks.on_agent_end(msg.content)
                if self.memory:
                    self.memory.save(self.history)
                print(f"[Agent] 最终回复: {msg.content[:200]}")
                print(f"{'='*60}\n")
                return msg.content

            # LLM 要求调用工具
            print(f"\n[Agent] → LLM 要求调用工具，准备执行...")
            self.history.append(msg.model_dump())
            self._handle_tool_calls(msg.tool_calls)

        print(f"\n[Agent] ✗ 已达到最大推理步数 ({self.max_iterations})，强制结束")
        if self.memory:
            self.memory.save(self.history)
        return "已达到最大推理步数，请简化问题后重试。"

    def invoke(self, user_message: str) -> str:
        """Runnable 协议接口，与 run() 等价。"""
        return self.run(user_message)

    def reset(self) -> None:
        """清空对话历史和记忆。"""
        self.history.clear()
        if self.memory:
            self.memory.clear()

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
                except json.JSONDecodeError as e:
                    content = f"[错误] 参数解析失败: {tc.function.arguments}"
                    self.callbacks.on_tool_error(tc.function.name, e)
                else:
                    self.callbacks.on_tool_start(tool.name, args)
                    try:
                        content = tool.execute(**args)
                    except Exception as e:
                        content = f"[错误] 工具执行失败: {e}"
                        self.callbacks.on_tool_error(tool.name, e)
                    else:
                        self.callbacks.on_tool_end(tool.name, content)

            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

        self.history.extend(results)


# ══════════════════════════════════════════════════════════════════════
# 以下为旧版 hooks 字典 + _emit() 方法的代码，已被 CallbackManager 替代。
# ══════════════════════════════════════════════════════════════════════

#     # 旧版 __init__ 参数:
#     #     hooks: dict[str, Callable] = None,
#     #     self.hooks = hooks or {}
#
#     def _emit(self, event: str, *args) -> None:
#         """[已废弃] 触发钩子回调。请使用 CallbackManager 替代。"""
#         if fn := self.hooks.get(event):
#             fn(*args)
