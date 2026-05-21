"""
Plan-and-Execute Agent —— 先规划再执行的两阶段 Agent。

与 ReAct Agent 的区别:
    ReAct: 思考 → 行动 → 观察 → 思考 → ...（交替进行）
    Plan-and-Execute: 制定完整计划 → 逐步执行 → 汇总结果（阶段分明）

适用场景:
    - 复杂多步任务（旅行规划、研究报告、项目分析）
    - 需要整体视野的任务（先规划可避免陷入局部最优）
    - 可分解为独立子任务的问题
"""

from agent.plan_execute.agent import PlanAndExecuteAgent

__all__ = ["PlanAndExecuteAgent"]
