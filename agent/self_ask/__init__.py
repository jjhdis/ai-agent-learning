"""
Self-Ask Agent —— 通过"追问→搜索→再追问"的循环解决复杂问题。

与 ReAct Agent 的区别:
    ReAct:      思考 → 调用工具 → 观察结果 → 思考 → ...（LLM 自主决定）
    Self-Ask:   Follow up(追问) → 搜索 → Intermediate answer(中间结果)
                → Follow up(再追问) → ... → So the final answer is(最终答案)

适用场景:
    - 多跳推理（需要链式查找多个信息）
    - 需要逐步缩小范围的问题
    - 使用不支持 Function Calling 的模型

核心区别 —— 纯文本模式:
    Self-Ask 不依赖 OpenAI 的 tools 参数，而是通过提示词工程让 LLM
    输出特定格式的文本（"Follow up:" / "So the final answer is:"），
    由 Agent 解析文本并执行搜索。这意味着它可以在任何 LLM 上运行。
"""

from agent.self_ask.agent import SelfAskAgent

__all__ = ["SelfAskAgent"]
