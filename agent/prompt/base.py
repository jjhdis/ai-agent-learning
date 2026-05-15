"""
PromptTemplate —— 字符串模板引擎。

支持 {variable} 变量替换、部分格式化（partial variables）、
以及 Runnable 协议兼容，可将模板作为可调用组件嵌入管道。
"""

import re
from string import Formatter
from typing import Any

from agent.chain.runnable import Runnable


class PromptTemplate(Runnable):
    """字符串提示词模板，用 {variable_name} 占位符标记变量。

    与 LangChain 的 PromptTemplate 设计一致：
    - format(**kwargs) 完成变量替换
    - 支持 partial() 生成部分填充的新模板
    - 实现 Runnable 协议，可在管道中作为组件使用

    Parameters:
        template: 模板字符串，如 "你好，{name}！今天{city}天气如何？"
        partial_variables: 预设的部分变量，在 format 时自动填入

    使用示例:
        tpl = PromptTemplate("你好，{name}！今天{city}天气如何？")
        result = tpl.format(name="小明", city="上海")
    """

    def __init__(self, template: str, partial_variables: dict[str, Any] = None):
        self.template = template
        self.partial_variables = dict(partial_variables) if partial_variables else {}
        self._input_variables = self._extract_variables(template)

    # ---- 只读属性 ----

    @property
    def input_variables(self) -> list[str]:
        """模板中尚未被 partial_variables 预设的变量名列表。"""
        return [v for v in self._input_variables if v not in self.partial_variables]

    # ---- 核心方法 ----

    def format(self, **kwargs) -> str:
        """用传入的关键字参数替换模板中的变量。

        Raises:
            KeyError: 有变量未提供值且未设置 partial_variables
        """
        merged = {**self.partial_variables, **kwargs}

        missing = [v for v in self._input_variables if v not in merged]
        if missing:
            raise KeyError(
                f"缺少模板变量: {missing}。"
                f"已提供: {list(kwargs.keys())}，"
                f"预设: {list(self.partial_variables.keys())}"
            )

        return self.template.format(**{v: merged[v] for v in self._input_variables})

    def partial(self, **kwargs) -> "PromptTemplate":
        """返回一个预设了部分变量的新模板。

        示例:
            tpl = PromptTemplate("城市: {city}, 日期: {date}")
            tpl2 = tpl.partial(city="上海")
            tpl2.format(date="2026-05-15")  # "城市: 上海, 日期: 2026-05-15"
        """
        new_partial = {**self.partial_variables, **kwargs}
        return PromptTemplate(self.template, new_partial)

    def invoke(self, input_data: Any) -> str:
        """Runnable 协议接口。

        当 input_data 是 dict 时，用 dict 中的值替换变量。
        其他类型直接作为 {input} 变量的值。
        """
        if isinstance(input_data, dict):
            return self.format(**input_data)
        return self.format(input=input_data)

    # ---- 私有 ----

    @staticmethod
    def _extract_variables(template: str) -> set[str]:
        """从模板字符串中提取所有 {variable_name} 占位符。"""
        return {f[1] for f in Formatter().parse(template) if f[1] is not None}

    def __repr__(self) -> str:
        vars_list = ", ".join(self._input_variables)
        return f"PromptTemplate(variables=[{vars_list}])"
