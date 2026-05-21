"""
XML Agent —— 使用 XML 标签格式调用工具的 Agent。

与 ReAct Agent 的区别:
    ReAct:      使用 OpenAI Function Calling 的 tools 参数（JSON Schema）
    XML Agent:  LLM 输出 XML 标签描述工具调用，Agent 解析 XML 后执行

格式对比:
    Function Calling:  tools=[{type:"function", function:{name:"search", ...}}]
                       → LLM 返回 tool_calls=[{function:{name:"search", arguments:{query:"..."}}}]

    XML Agent:         提示词中描述工具 → LLM 输出 <tool_call><name>search</name>
                       <arguments>{"query":"..."}</arguments></tool_call>
                       → Agent 解析 XML 执行工具 → 反馈 <tool_result>...</tool_result>

适用场景:
    - 使用不支持 Function Calling 的模型（如一些开源模型）
    - 需要对工具调用格式有更多控制权的场景
    - 调试时更直观（XML 标签比 JSON 更易读）
"""

from agent.xml_agent.agent import XMLAgent

__all__ = ["XMLAgent"]
