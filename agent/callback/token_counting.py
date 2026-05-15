"""
TokenCountingCallback —— Token 消耗统计回调处理器。

统计每次 Agent 运行的 Token 消耗（输入 Token、输出 Token、总 Token），
可用于监控 API 费用和优化提示词长度。

支持自定义 Token 单价，自动计算预估费用。
"""

from agent.callback.base import BaseCallbackHandler


# 常见模型的 Token 单价（单位：元/百万 Token）
# 数据来源：各模型官方定价（2025年）
DEFAULT_PRICES = {
    # DeepSeek
    "deepseek-chat": {"input": 1.0, "output": 2.0},
    "deepseek-reasoner": {"input": 4.0, "output": 16.0},
    # OpenAI
    "gpt-4o": {"input": 15.0, "output": 60.0},
    "gpt-4o-mini": {"input": 1.5, "output": 6.0},
    "gpt-3.5-turbo": {"input": 3.0, "output": 6.0},
    # 通义千问
    "qwen-plus": {"input": 2.0, "output": 6.0},
    "qwen-max": {"input": 20.0, "output": 60.0},
    # GLM
    "glm-4-plus": {"input": 5.0, "output": 5.0},
}


class TokenCountingCallback(BaseCallbackHandler):
    """统计 Agent 运行过程中的 Token 消耗。

    统计指标：
        - prompt_tokens: 输入 Token 总数（所有 LLM 调用累加）
        - completion_tokens: 输出 Token 总数
        - total_tokens: 总 Token 数
        - llm_call_count: LLM 调用次数
        - estimated_cost: 预估费用（元）

    Parameters:
        model_name: 模型名称，用于计算费用。为 None 时从 response 中自动获取。
        prices: 自定义价格表，覆盖 DEFAULT_PRICES。格式：{"模型名": {"input": 单价, "output": 单价}}
        verbose: 是否在 Agent 结束时打印统计摘要，默认 True。
    """

    def __init__(
        self,
        model_name: str = None,
        prices: dict[str, dict[str, float]] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.prices = {**DEFAULT_PRICES, **(prices or {})}
        self.verbose = verbose

        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._llm_call_count: int = 0
        self._actual_model: str = ""

    # ---- 只读统计属性 ----

    @property
    def prompt_tokens(self) -> int:
        """累计输入 Token 数。"""
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """累计输出 Token 数。"""
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        """累计总 Token 数。"""
        return self._total_tokens

    @property
    def llm_call_count(self) -> int:
        """LLM 调用次数。"""
        return self._llm_call_count

    @property
    def estimated_cost(self) -> float:
        """预估费用（元），基于实际使用的模型和 Token 单价计算。"""
        model = self._actual_model or self.model_name or "deepseek-chat"
        price = self.prices.get(model, {"input": 1.0, "output": 2.0})
        input_cost = self._prompt_tokens * price["input"] / 1_000_000
        output_cost = self._completion_tokens * price["output"] / 1_000_000
        return round(input_cost + output_cost, 6)

    def reset(self) -> None:
        """重置所有统计计数器。"""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._llm_call_count = 0
        self._actual_model = ""

    # ---- 事件处理 ----

    def on_agent_start(self, message: str) -> None:
        """Agent 启动时重置计数器。"""
        self.reset()

    def on_llm_end(self, response) -> None:
        """LLM 调用完成时，从 response 中提取 usage 信息并累加。

        response 是 OpenAI ChatCompletion 对象，包含 .usage 属性：
            - usage.prompt_tokens: 本次输入 Token 数
            - usage.completion_tokens: 本次输出 Token 数
            - usage.total_tokens: 本次总 Token 数
        """
        self._llm_call_count += 1

        # 记录实际使用的模型名（取第一次调用的模型）
        if not self._actual_model and hasattr(response, "model"):
            self._actual_model = response.model

        # 提取 usage 信息
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        self._prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self._completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self._total_tokens += getattr(usage, "total_tokens", 0) or 0

        if self.verbose:
            model_display = self._actual_model or "unknown"
            print(
                f"[Token] LLM #{self._llm_call_count} — "
                f"模型: {model_display}, "
                f"输入: {getattr(usage, 'prompt_tokens', 0):,} tokens, "
                f"输出: {getattr(usage, 'completion_tokens', 0):,} tokens, "
                f"本次合计: {getattr(usage, 'total_tokens', 0):,} tokens"
            )

    def on_agent_end(self, reply: str) -> None:
        """Agent 完成时打印统计摘要。"""
        if not self.verbose:
            return

        model = self._actual_model or self.model_name or "unknown"
        print(
            f"[Token] ════════════════════════════════════════════\n"
            f"[Token]  Token 消耗统计\n"
            f"[Token]  ════════════════════════════════════════════\n"
            f"[Token]  模型: {model}\n"
            f"[Token]  LLM 调用次数: {self._llm_call_count}\n"
            f"[Token]  输入 Token:   {self._prompt_tokens:>8,}\n"
            f"[Token]  输出 Token:   {self._completion_tokens:>8,}\n"
            f"[Token]  总 Token:     {self._total_tokens:>8,}\n"
            f"[Token]  预估费用:     ¥{self.estimated_cost:.6f}\n"
            f"[Token]  ════════════════════════════════════════════"
        )

    def on_agent_error(self, error: Exception) -> None:
        """Agent 出错时也打印已消耗的 Token 统计。"""
        if self.verbose and self._llm_call_count > 0:
            print(
                f"[Token] Agent 出错，已消耗 Token: "
                f"输入 {self._prompt_tokens:,} / "
                f"输出 {self._completion_tokens:,} / "
                f"总计 {self._total_tokens:,}"
            )
