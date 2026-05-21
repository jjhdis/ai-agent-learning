"""
XML Agent —— 使用 XML 标签格式调用工具的 Agent。

核心理念:
    不依赖 OpenAI Function Calling（tools 参数），而是通过提示词工程
    让 LLM 以 XML 格式输出工具调用指令。Agent 解析 XML、执行工具、
    将结果用 XML 标签包裹反馈给 LLM，循环直到 LLM 给出最终文本回复。

适用场景:
    - 模型不支持 Function Calling（如 Llama、Mistral 等开源模型）
    - 需要对工具调用做更细粒度的控制
    - 调试阶段 —— XML 比 JSON Schema 更直观可读
"""

import re
import json

from agent.chain.runnable import Runnable
from agent.llm.client import LLMClient


class XMLAgent(Runnable):
    """XML 格式的 Agent —— 通过 <tool_call> / <tool_result> 标签与 LLM 交互。

    执行流程:
        1. 将任务 + 系统提示词（含工具描述和 XML 格式规范）发给 LLM
        2. LLM 返回包含 <tool_call>...</tool_call> 的文本
        3. Agent 解析 XML 提取工具名和参数 → 执行工具
        4. Agent 将结果包装为 <tool_result>...</tool_result> 追加到对话
        5. LLM 看到结果后继续推理或给出最终答案
        6. 重复 2-5，直到 LLM 输出不含 <tool_call> 的回复

    与 ReAct Agent 的关键区别:
        ReAct:  tools 参数通过 OpenAI API 传递 → LLM 返回结构化 tool_calls
        XML:    工具信息写在 System Prompt 里 → LLM 返回带标签的自然语言文本

    Parameters:
        llm_client: LLM 客户端
        tools: 工具描述列表，格式为 [{"name": "xxx", "description": "xxx", "parameters": {...}}, ...]
               如果为 None，则不使用工具，LLM 直接回答
        max_iterations: 最多工具调用轮数，防止无限循环
        verbose: 是否打印 XML 解析和执行过程
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: list[dict] = None,
        tool_executor=None,
        max_iterations: int = 5,
        verbose: bool = True,
    ):
        self.llm = llm_client
        self.tools = tools or []
        self.tool_executor = tool_executor or (lambda name, args: f"[工具 '{name}' 未注册]")
        self.max_iterations = max_iterations
        self.verbose = verbose

        # 缓存构建好的系统提示词
        self._system_prompt = _build_xml_system_prompt(self.tools)

    # ---- 公共 API ----

    def run(self, task: str) -> str:
        """处理一个任务，通过 XML 标签驱动工具调用。

        Args:
            task: 用户任务描述

        Returns:
            最终回答字符串
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
        ]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[XMLAgent] 任务: {task}")
            print(f"[XMLAgent] 可用工具: {[t['name'] for t in self.tools]}")
            print(f"{'='*60}")

        # 第一轮：用户任务
        messages.append({"role": "user", "content": task})

        for i in range(self.max_iterations):
            response = self.llm.chat(messages)
            text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": text})

            if self.verbose:
                print(f"\n[XMLAgent] 第 {i+1} 轮 LLM 输出:")
                # 只显示 XML 相关的片段
                if "<tool_call>" in text:
                    for tag in _extract_tool_calls(text):
                        print(f"  -> tool_call: {tag['name']}({tag['arguments']})")
                else:
                    print(f"  {text[:200]}")

            # 解析 XML 工具调用
            tool_calls = _extract_tool_calls(text)

            if not tool_calls:
                # 没有工具调用 → 这就是最终回复
                if self.verbose:
                    print(f"\n[XMLAgent] LLM 给出最终回复（无工具调用）")
                return text

            # 执行工具调用
            tool_results = []
            for tc in tool_calls:
                name = tc["name"]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"].strip() else {}
                except json.JSONDecodeError:
                    args = {"query": tc["arguments"]}

                if self.verbose:
                    print(f"\n[XMLAgent] 执行工具: {name}({args})")

                try:
                    result = self.tool_executor(name, args)
                except Exception as e:
                    result = f"[错误] 工具执行失败: {e}"

                if self.verbose:
                    preview = result[:150] + ("..." if len(result) > 150 else "")
                    print(f"[XMLAgent] 工具结果: {preview}")

                tool_results.append(_format_tool_result(name, result))

            # 将工具结果追加到对话
            results_text = "\n".join(tool_results)
            messages.append({"role": "user", "content": results_text})

        # 达到最大轮数
        if self.verbose:
            print(f"\n[XMLAgent] 达到最大轮数，要求最终回答")
        messages.append(
            {"role": "user", "content": "请基于以上所有工具结果，给出最终回答。"}
        )
        response = self.llm.chat(messages)
        return response.choices[0].message.content

    def invoke(self, task: str) -> str:
        return self.run(task)


# ══════════════════════════════════════════════════════════════════════
# XML 解析辅助函数
# ══════════════════════════════════════════════════════════════════════

def _extract_tool_calls(text: str) -> list[dict]:
    """从文本中提取所有 <tool_call> 标签的内容。

    支持的格式:
        <tool_call>
        <name>工具名</name>
        <arguments>{"key": "value"}</arguments>
        </tool_call>

    返回:
        [{"name": "工具名", "arguments": '{"key": "value"}'}, ...]
    """
    pattern = (
        r"<tool_call>\s*"
        r"<name>(.*?)</name>\s*"
        r"<arguments>(.*?)</arguments>\s*"
        r"</tool_call>"
    )
    matches = re.findall(pattern, text, re.DOTALL)
    return [
        {"name": name.strip(), "arguments": args.strip()}
        for name, args in matches
    ]


def _format_tool_result(name: str, result: str) -> str:
    """将工具执行结果格式化为 XML 标签文本。"""
    return (
        f"<tool_result>\n"
        f"<name>{name}</name>\n"
        f"<result>{result}</result>\n"
        f"</tool_result>"
    )


def _build_xml_system_prompt(tools: list[dict]) -> str:
    """构建包含工具描述和 XML 格式说明的系统提示词。"""
    if not tools:
        return "你是一个有帮助的 AI 助手。请直接回答用户的问题。"

    tools_desc_parts = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "无描述")
        params = tool.get("parameters", {})
        params_desc = json.dumps(params, ensure_ascii=False, indent=2)
        tools_desc_parts.append(
            f"  - {name}: {desc}\n"
            f"    参数格式: {params_desc}"
        )
    tools_desc = "\n".join(tools_desc_parts)

    return f"""你是一个 AI 助手，可以使用以下工具来完成任务。

## 可用工具
{tools_desc}

## 工具调用格式

当你需要使用工具时，请严格使用以下 XML 格式（可以一次调用多个工具）：

<tool_call>
<name>工具名称</name>
<arguments>JSON 格式的参数</arguments>
</tool_call>

## 工具结果格式

调用工具后，你会收到以下格式的结果：

<tool_result>
<name>工具名称</name>
<result>执行结果</result>
</tool_result>

## 规则
1. 需要工具时就输出 <tool_call> 标签，可以一次输出多个
2. arguments 必须是合法的 JSON 字符串
3. 当你已经获取足够信息时，直接输出最终回答（不要使用 <tool_call> 标签）
4. 基于工具结果回答，不要编造信息"""
