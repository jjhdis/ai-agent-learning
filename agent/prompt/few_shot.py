"""
Few-shot 示例管理系统。

提供:
    - BaseExampleSelector: 示例选择器抽象基类
    - LengthBasedExampleSelector: 基于长度的示例选择器
    - FewShotPromptTemplate: 将示例动态插入提示词的模板

与 LangChain 的 FewShotPromptTemplate + ExampleSelector 设计一致。
"""

from abc import ABC, abstractmethod
from typing import Any

from agent.chain.runnable import Runnable
from agent.prompt.base import PromptTemplate


# ──────────────────────────────────────────────
# 示例选择器
# ──────────────────────────────────────────────

class BaseExampleSelector(ABC):
    """示例选择器抽象基类。

    负责从示例池中选出一组最合适的示例插入到提示词中。
    """

    @abstractmethod
    def select_examples(self, input_variables: dict[str, Any]) -> list[dict]:
        """根据当前输入变量选择一组示例。"""
        ...

    @abstractmethod
    def add_example(self, example: dict) -> None:
        """向示例池中添加一个示例。"""
        ...


class LengthBasedExampleSelector(BaseExampleSelector):
    """基于 Token 总长度的示例选择器。

    从示例池中选择尽可能多的示例，但不超过 max_tokens 长度限制。
    越晚加入的示例优先级越高（认为更相关）。

    Parameters:
        examples: 初始示例列表，每个示例是一个 dict
        example_prompt: 用于格式化单个示例的模板
        max_tokens: 示例区域的 Token 上限（用字符数近似，中文字符 ≈1 token，英文单词 ≈1-2 token）

    使用示例:
        selector = LengthBasedExampleSelector(
            examples=[
                {"city": "北京", "weather": "晴，25°C"},
                {"city": "上海", "weather": "多云，28°C"},
            ],
            example_prompt=PromptTemplate("城市: {city}, 天气: {weather}"),
            max_tokens=100,
        )
        selected = selector.select_examples({"city": "广州"})
    """

    def __init__(
        self,
        examples: list[dict],
        example_prompt: PromptTemplate,
        max_tokens: int = 200,
    ):
        self._examples = list(examples)
        self.example_prompt = example_prompt
        self.max_tokens = max_tokens

    def select_examples(self, input_variables: dict[str, Any]) -> list[dict]:
        """选择不超过 max_tokens 长度的示例子集。"""
        selected: list[dict] = []
        current_length = 0

        # 从后向前遍历（越新的示例优先级越高）
        for example in reversed(self._examples):
            try:
                formatted = self.example_prompt.format(**example)
            except KeyError:
                continue
            new_length = current_length + len(formatted)
            if new_length > self.max_tokens and selected:
                break
            selected.insert(0, example)
            current_length = new_length

        return selected

    def add_example(self, example: dict) -> None:
        """添加一个新示例到池中。"""
        self._examples.append(example)

    def __repr__(self) -> str:
        return f"LengthBasedExampleSelector(examples={len(self._examples)}, max_tokens={self.max_tokens})"


# ──────────────────────────────────────────────
# Few-Shot 模板
# ──────────────────────────────────────────────

class FewShotPromptTemplate(Runnable):
    """Few-shot 提示词模板，运行时从示例池中选择示例并插入。

    模板结构:
        prefix（前缀说明）
        + 示例（根据当前输入动态选择）
        + suffix（后缀 / 最终问题）

    Parameters:
        example_selector: 示例选择器
        example_prompt: 格式化单个示例的模板
        prefix: 示例前的说明文字（可包含变量）
        suffix: 示例后的用户问题（可包含变量）
        example_separator: 示例之间的分隔符

    使用示例:
        few_shot = FewShotPromptTemplate(
            example_selector=selector,
            example_prompt=PromptTemplate("输入: {input}\n回答: {answer}"),
            prefix="以下是一些对话示例:\n",
            suffix="\n现在请回答: {input}",
        )
        result = few_shot.format(input="上海天气如何？")
    """

    def __init__(
        self,
        example_selector: BaseExampleSelector,
        example_prompt: PromptTemplate,
        prefix: str = "",
        suffix: str = "",
        example_separator: str = "\n\n",
    ):
        self.example_selector = example_selector
        self.example_prompt = example_prompt
        self.prefix = prefix
        self.suffix = suffix
        self.example_separator = example_separator

    # ---- 核心方法 ----

    def format(self, **kwargs) -> str:
        """选择示例并拼装完整提示词。"""
        examples = self.example_selector.select_examples(kwargs)

        parts: list[str] = []

        # 前缀
        if self.prefix:
            parts.append(self.prefix.format(**kwargs) if "{" in self.prefix else self.prefix)

        # 示例
        for ex in examples:
            try:
                parts.append(self.example_prompt.format(**ex))
            except KeyError:
                continue

        # 后缀
        if self.suffix:
            parts.append(self.suffix.format(**kwargs) if "{" in self.suffix else self.suffix)

        return self.example_separator.join(parts) if len(parts) > 1 else (parts[0] if parts else "")

    def invoke(self, input_data: Any) -> str:
        """Runnable 协议接口。"""
        if isinstance(input_data, dict):
            return self.format(**input_data)
        return self.format(input=input_data)

    def __repr__(self) -> str:
        return (
            f"FewShotPromptTemplate("
            f"selector={self.example_selector}, "
            f"prefix_len={len(self.prefix)}, "
            f"suffix_len={len(self.suffix)})"
        )
