"""
Runnable 协议 —— 可组合组件的统一接口。

任何实现 .invoke() 的组件都可以通过 | 运算符自由组合，
类似 LangChain 的 LCEL（LangChain Expression Language）。
"""

from abc import ABC, abstractmethod
from typing import Any


class Runnable(ABC):
    """可运行组件的统一接口。

    Parameters:
        input_data: 任意类型的输入数据

    子类只需实现 invoke()，即可自动获得组合能力。
    """

    @abstractmethod
    def invoke(self, input_data: Any) -> Any:
        """处理输入并返回输出。"""
        ...

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """用 | 运算符串联两个 Runnable。

        返回一个 RunnableSequence，self 的输出作为 other 的输入。
        """
        if not isinstance(other, Runnable):
            raise TypeError(f"右侧操作数必须实现 Runnable 协议，得到: {type(other)}")
        steps: list[Runnable] = []
        if isinstance(self, RunnableSequence):
            steps.extend(self.steps)
        else:
            steps.append(self)
        if isinstance(other, RunnableSequence):
            steps.extend(other.steps)
        else:
            steps.append(other)
        return RunnableSequence(steps)

    def batch(self, inputs: list[Any]) -> list[Any]:
        """批量处理多个输入。"""
        return [self.invoke(inp) for inp in inputs]


class RunnableSequence(Runnable):
    """由 | 运算符自动创建的串行执行链。

    每一步的输出作为下一步的输入，形成处理管道。
    """

    def __init__(self, steps: list[Runnable]):
        self.steps = steps

    def invoke(self, input_data: Any) -> Any:
        result = input_data
        for step in self.steps:
            result = step.invoke(result)
        return result

    def __repr__(self) -> str:
        steps_repr = " | ".join(type(s).__name__ for s in self.steps)
        return f"RunnableSequence({steps_repr})"
