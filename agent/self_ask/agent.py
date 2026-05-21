"""
Self-Ask Agent —— 纯文本驱动的追问-搜索-回答循环。

核心理念:
    不依赖 OpenAI Function Calling，而是通过提示词工程让 LLM 输出
    特定格式的文本。Agent 解析文本、执行搜索、将结果反馈给 LLM，
    循环直到 LLM 输出最终答案。

原始论文: "Measuring and Narrowing the Compositionality Gap in Language Models"
"""

from typing import Optional

from agent.chain.runnable import Runnable
from agent.llm.client import LLMClient


class SelfAskAgent(Runnable):
    """Self-Ask Agent —— 通过 Follow up / So the final answer is 模式推理。

    执行流程:
        1. 将用户问题 + 系统提示词发给 LLM
        2. LLM 返回 "Follow up: <子问题>" → Agent 搜索并反馈 "Intermediate answer: <结果>"
        3. LLM 继续追问或返回 "So the final answer is: <答案>"
        4. 重复步骤 2-3，最多 max_follow_ups 轮

    Parameters:
        llm_client: LLM 客户端
        search_fn: 搜索函数，接收查询字符串，返回结果字符串。
                   不传则使用 LLM 直接回答（无工具模式）。
        max_follow_ups: 最多追问次数，防止无限循环
        verbose: 是否打印追问和搜索过程
    """

    def __init__(
        self,
        llm_client: LLMClient,
        search_fn=None,
        max_follow_ups: int = 5,
        verbose: bool = True,
    ):
        self.llm = llm_client
        self.search_fn = search_fn or (lambda q: "[无可用的搜索工具]")
        self.max_follow_ups = max_follow_ups
        self.verbose = verbose

    # ---- 公共 API ----

    def run(self, question: str) -> str:
        """处理一个复杂问题，通过追问→搜索→追问的循环逐步解决。

        Args:
            question: 用户问题（越复杂越能体现 Self-Ask 的优势）

        Returns:
            最终答案字符串
        """
        # 构建初始上下文
        context = f"Question: {question}\n"
        messages = [
            {"role": "system", "content": _SELF_ASK_SYSTEM_PROMPT},
        ]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SelfAsk] 问题: {question}")
            print(f"{'='*60}")

        for i in range(self.max_follow_ups):
            messages.append({"role": "user", "content": context})
            response = self.llm.chat(messages)
            text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": text})

            if self.verbose:
                print(f"\n[SelfAsk] 第 {i+1} 轮 LLM 输出:")
                print(f"  {text[:300]}")

            # 检查是否给出了最终答案
            final_answer = _extract_final_answer(text)
            if final_answer:
                if self.verbose:
                    print(f"\n[SelfAsk] 得出最终答案")
                return final_answer

            # 检查是否有追问
            follow_up = _extract_follow_up(text)
            if follow_up:
                if self.verbose:
                    print(f"\n[SelfAsk] 追问: {follow_up}")
                    print(f"[SelfAsk] 搜索中...")

                search_result = self.search_fn(follow_up)

                if self.verbose:
                    print(f"[SelfAsk] 搜索完成，结果长度: {len(search_result)} 字符")

                context += (
                    f"{text}\n"
                    f"Intermediate answer: {search_result}\n"
                )
            else:
                # LLM 既没有追问也没有给出最终答案 —— 强制要求继续
                if self.verbose:
                    print(f"\n[SelfAsk] LLM 未输出预期格式，要求继续")
                context += f"{text}\n(请继续: 追问或给出最终答案)\n"

        # 达到最大追问次数，强制要求 LLM 给出最终答案
        if self.verbose:
            print(f"\n[SelfAsk] 达到最大追问次数，强制要求最终答案")
        context += "\n请基于以上信息给出最终答案。\n"
        messages.append({"role": "user", "content": context})
        response = self.llm.chat(messages)
        text = response.choices[0].message.content
        final = _extract_final_answer(text)
        return final or text

    def invoke(self, question: str) -> str:
        return self.run(question)


# ══════════════════════════════════════════════════════════════════════
# 解析辅助函数
# ══════════════════════════════════════════════════════════════════════

def _extract_final_answer(text: str) -> Optional[str]:
    """从 LLM 输出中提取最终答案。

    匹配模式: "So the final answer is:" 之后的内容，
    或者 "Final answer:" / "最终答案:" 之后的内容。
    """
    import re

    patterns = [
        r"So the final answer is:\s*(.*)",
        r"Final answer:\s*(.*)",
        r"最终答案[：:]\s*(.*)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _extract_follow_up(text: str) -> Optional[str]:
    """从 LLM 输出中提取追问查询。

    匹配模式: "Follow up:" 之后的内容。
    """
    import re

    m = re.search(r"Follow up:\s*(.*)", text, re.IGNORECASE)
    if m:
        follow_up = m.group(1).strip()
        # 清理常见的前后缀噪声
        follow_up = re.sub(r"^['\"]+|['\"]+$", "", follow_up)
        return follow_up if len(follow_up) > 3 else None
    return None


# ══════════════════════════════════════════════════════════════════════
# System Prompt
# ══════════════════════════════════════════════════════════════════════

_SELF_ASK_SYSTEM_PROMPT = """你是一个 Self-Ask Agent。对于复杂问题，你需要通过追问来逐步获取信息。

请严格按照以下格式回复:

当需要查找信息时:
Follow up: <具体的搜索查询>

当你已经获取了足够的信息，可以回答用户问题时:
So the final answer is: <完整、详细的答案>

规则:
1. 每次只能追问一个问题（一次只写一行 "Follow up:"）
2. 追问应该是具体的搜索查询，而不是模糊的问题
3. 基于所有 "Intermediate answer:" 中的信息来回答问题
4. 在给出最终答案前，确保已获取所有必要信息
5. 永远不要编造信息 —— 只使用 Intermediate answer 中提供的事实"""
