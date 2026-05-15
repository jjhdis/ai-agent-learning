"""
ChatPromptTemplate —— 多轮对话模板。

支持 system / user / assistant 多角色消息模板，
以及 MessagePlaceholder 在运行时插入消息列表。
与 LangChain 的 ChatPromptTemplate 设计一致。
"""

from typing import Any

from agent.chain.runnable import Runnable
from agent.prompt.base import PromptTemplate


class ChatPromptTemplate(Runnable):
    """多轮对话提示词模板。

    每个模板项可以是:
        - (role, template_string) 元组: role 为 "system" / "user" / "assistant"
        - MessagePlaceholder: 运行时插入一个消息列表

    Parameters:
        messages: 模板消息列表

    使用示例:
        tpl = ChatPromptTemplate([
            ("system", "你是一个{role}助手，擅长{skill}"),
            ("user", "{input}"),
        ])
        msgs = tpl.format_messages(role="天气", skill="查询天气", input="上海天气")
        # → [
        #     {"role": "system", "content": "你是一个天气助手，擅长查询天气"},
        #     {"role": "user", "content": "上海天气"},
        # ]
    """

    def __init__(self, messages: list):
        self._messages = messages

    # ---- 核心方法 ----

    def format_messages(self, **kwargs) -> list[dict]:
        """用传入的变量替换所有模板，返回消息列表。

        对于 MessagePlaceholder 项，从 kwargs 中取出对应的消息列表并扁平插入。
        """
        result: list[dict] = []
        for item in self._messages:
            if isinstance(item, tuple):
                role, template_str = item
                tpl = PromptTemplate(template_str)
                content = tpl.format(**kwargs)
                result.append({"role": role, "content": content})
            else:
                # MessagePlaceholder 或任何有 variable_name 的对象
                var_name = getattr(item, "variable_name", None)
                if var_name and var_name in kwargs:
                    messages = kwargs[var_name]
                    if isinstance(messages, list):
                        result.extend(messages)
        return result

    def invoke(self, input_data: Any) -> list[dict]:
        """Runnable 协议接口。

        当 input_data 是 dict 时，用 dict 中的值替换变量。
        """
        if isinstance(input_data, dict):
            return self.format_messages(**input_data)
        return self.format_messages(input=input_data)

    # ---- 工厂方法 ----

    @classmethod
    def from_messages(cls, messages: list) -> "ChatPromptTemplate":
        """工厂方法，与 LangChain 的 ChatPromptTemplate.from_messages() 保持一致。

        示例:
            ChatPromptTemplate.from_messages([
                ("system", "你是一个{role}助手"),
                ("user", "{input}"),
            ])
        """
        return cls(messages)

    def __repr__(self) -> str:
        items = []
        for item in self._messages:
            if isinstance(item, tuple):
                items.append(f"{item[0]}: {item[1][:30]}...")
            else:
                items.append(repr(item))
        return f"ChatPromptTemplate({', '.join(items)})"
