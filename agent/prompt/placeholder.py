"""
MessagePlaceholder —— 运行时消息列表占位符。

在 ChatPromptTemplate 中标记一个位置，该位置将在运行时
由外部传入的消息列表填充。

与 LangChain 的 MessagesPlaceholder 设计一致。
"""


class MessagePlaceholder:
    """占位符，用于在 ChatPromptTemplate 中插入可变长度的消息列表。

    Parameters:
        variable_name: 占位符变量名，format_messages() 时从 kwargs 中取值

    使用示例:
        # 创建带占位符的对话模板
        tpl = ChatPromptTemplate([
            ("system", "你是一个AI助手"),
            MessagePlaceholder("history"),     # ← 运行时插入历史消息
            ("user", "{input}"),
        ])

        # 运行时填充
        msgs = tpl.format_messages(
            input="帮我查天气",
            history=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮你的？"},
            ]
        )
    """

    def __init__(self, variable_name: str):
        self.variable_name = variable_name

    def __repr__(self) -> str:
        return f"MessagePlaceholder({self.variable_name})"
