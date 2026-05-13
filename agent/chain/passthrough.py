"""
数据传递与变换组件。

提供:
    - RunnablePassthrough: 原样透传输入
    - RunnableMap: 将多个 Runnable 的输出合并为 dict
    - RunnableLambda: 包装普通函数为 Runnable
"""

from typing import Any, Callable

from agent.chain.runnable import Runnable


class RunnablePassthrough(Runnable):
    """原样透传输入，不做任何处理。

    常用于 LCEL 管道中需要保持数据原始形态的位置。
    """

    def invoke(self, input_data: Any) -> Any:
        return input_data


class RunnableMap(Runnable):
    """并行执行多个 Runnable，将输出合并为 dict。

    类似 LangChain 的 RunnableParallel，
    用于将同一输入分流到不同的处理分支。

    示例:
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough(),
        })
    """

    def __init__(self, mapping: dict[str, Runnable]):
        self.mapping = mapping

    def invoke(self, input_data: Any) -> dict[str, Any]:
        return {key: r.invoke(input_data) for key, r in self.mapping.items()}


class RunnableLambda(Runnable):
    """将普通函数包装成 Runnable。

    示例:
        RunnableLambda(lambda x: x.upper()).invoke("hello")  # "HELLO"
    """

    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)
