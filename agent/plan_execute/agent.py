"""
Plan-and-Execute Agent —— 先规划再执行的两阶段 Agent。

生命周期:
    User 任务 → [阶段1: Planner 制定计划] → [阶段2: Executor 逐步执行]
    → [阶段3: Summarizer 汇总结果] → 最终回复

与现有 Agent 的关系:
    PlanAndExecuteAgent 是高层编排器，它内部为每个执行步骤创建新的 Agent
    实例（ReAct 模式）来完成具体工作。相当于"总指挥 + 多个执行者"的架构。
"""

from agent.chain.runnable import Runnable
from agent.llm.client import LLMClient
from agent.tools.registry import ToolRegistry
from agent.callback.manager import CallbackManager
from agent.core.agent import Agent


class PlanAndExecuteAgent(Runnable):
    """先规划再执行的两阶段 Agent。

    流程:
        Phase 1 - Planner:  LLM 分析任务，生成有序步骤列表
        Phase 2 - Executor: 逐步执行，每步创建一个 ReAct Agent，
                             自动携带之前步骤的结果作为上下文
        Phase 3 - Summarizer: LLM 汇总所有执行结果，生成最终回答

    Parameters:
        llm_client: LLM 客户端（与 Agent 共用同一个）
        registry: 工具注册中心，规划时展示可用工具，执行时分发工具
        callbacks: 回调管理器，监听规划/执行/汇总各阶段事件
        max_plan_steps: 计划最多包含的步骤数（防止过度拆分）
        verbose: 是否打印规划的步骤和执行进度
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ToolRegistry = None,
        callbacks: CallbackManager = None,
        max_plan_steps: int = 8,
        verbose: bool = True,
    ):
        self.llm = llm_client
        self.registry = registry or ToolRegistry()
        self.callbacks = callbacks or CallbackManager()
        self.max_plan_steps = max_plan_steps
        self.verbose = verbose

    # ---- 公共 API ----

    def run(self, task: str) -> str:
        """执行完整的 Plan → Execute → Summarize 流程。

        Args:
            task: 用户任务描述，越具体规划质量越高

        Returns:
            汇总后的最终回答（字符串）
        """
        self.callbacks.on_agent_start(task)

        # Phase 1: 制定计划
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[PlanAndExecute] [Phase 1] 制定计划")
            print(f"{'='*60}")
            print(f"[PlanAndExecute] 任务: {task}")

        plan = self._generate_plan(task)
        if not plan:
            fallback = self._fallback_execute(task)
            self.callbacks.on_agent_end(fallback)
            return fallback

        if self.verbose:
            print(f"\n[PlanAndExecute] 计划共 {len(plan)} 步:")
            for i, step in enumerate(plan, 1):
                print(f"    {i}. {step}")

        # Phase 2: 逐步执行
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[PlanAndExecute] [Phase 2] 逐步执行")
            print(f"{'='*60}")

        results = self._execute_plan(plan)

        # Phase 3: 汇总结果
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[PlanAndExecute] [Phase 3] 汇总结果")
            print(f"{'='*60}")

        final_answer = self._summarize(task, plan, results)
        self.callbacks.on_agent_end(final_answer)
        return final_answer

    def invoke(self, task: str) -> str:
        """Runnable 协议接口，与 run() 等价。"""
        return self.run(task)

    # ---- Phase 1: Planner ----

    def _generate_plan(self, task: str) -> list[str]:
        """调用 LLM 为任务生成分步执行计划。

        提示词设计要点:
        - 告知可用工具有哪些（让规划者知道能力边界）
        - 要求输出编号列表（便于解析）
        - 不要求解释（减少无关输出）
        """
        tools_desc = self._describe_tools()
        prompt = _PLAN_SYSTEM_PROMPT.format(tools=tools_desc)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"请为以下任务制定执行计划:\n\n{task}"},
        ]

        self.callbacks.on_llm_start(messages, None)
        try:
            response = self.llm.chat(messages)
        except Exception as e:
            self.callbacks.on_llm_error(e)
            return []

        content = response.choices[0].message.content
        self.callbacks.on_llm_end(response)
        self.callbacks.on_think(content)

        plan = self._parse_plan(content)
        return plan[: self.max_plan_steps]

    # ---- Phase 2: Executor ----

    def _execute_plan(self, plan: list[str]) -> list[dict]:
        """逐步执行计划，每步创建一个新的 ReAct Agent。

        每个步骤会收到:
        - 步骤描述本身
        - 之前所有步骤的执行结果（作为上下文）

        使用独立的 Agent 实例确保每步的历史隔离，
        避免上一步的 ReAct 内部循环污染下一步的上下文。
        """
        results = []
        context_parts: list[str] = []

        for i, step in enumerate(plan, 1):
            if self.verbose:
                print(f"\n{'─'*50}")
                print(f"[PlanAndExecute] 执行步骤 {i}/{len(plan)}: {step}")
                print(f"{'─'*50}")

            # 构建带上下文的步骤描述
            if context_parts:
                context_block = "\n".join(context_parts)
                full_step = (
                    f"{step}\n\n"
                    f"【之前步骤的结果 —— 请基于这些信息继续】\n"
                    f"{'─'*40}\n{context_block}\n{'─'*40}"
                )
            else:
                full_step = step

            # 为每一步创建独立的 Agent（隔离 ReAct 循环历史）
            executor = Agent(
                llm_client=self.llm,
                registry=self.registry,
                callbacks=self.callbacks,
            )
            try:
                result = executor.run(full_step)
                result_str = str(result)
            except Exception as e:
                result_str = f"[步骤执行失败] {e}"

            results.append({"step": step, "result": result_str})
            context_parts.append(f"步骤{i}: {step}\n结果: {result_str}")

        return results

    # ---- Phase 3: Summarizer ----

    def _summarize(self, task: str, plan: list[str], results: list[dict]) -> str:
        """汇总所有步骤的执行结果，生成最终回答。"""
        plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))

        results_text_parts = []
        for i, r in enumerate(results, 1):
            results_text_parts.append(f"步骤{i}: {r['step']}\n结果: {r['result']}")
        results_text = "\n\n".join(results_text_parts)

        prompt = _SUMMARIZE_PROMPT.format(
            task=task, plan=plan_text, results=results_text
        )

        messages = [{"role": "user", "content": prompt}]

        self.callbacks.on_llm_start(messages, None)
        try:
            response = self.llm.chat(messages)
        except Exception as e:
            self.callbacks.on_llm_error(e)
            return "\n\n".join(f"{i}. {r['step']}\n   → {r['result']}"
                               for i, r in enumerate(results, 1))

        content = response.choices[0].message.content
        self.callbacks.on_llm_end(response)
        return content

    # ---- 辅助方法 ----

    def _fallback_execute(self, task: str) -> str:
        """计划生成失败时的降级方案：直接用 ReAct Agent 执行。"""
        if self.verbose:
            print("[PlanAndExecute] [WARN] 计划生成失败，降级为直接执行")

        executor = Agent(
            llm_client=self.llm,
            registry=self.registry,
            callbacks=self.callbacks,
        )
        return executor.run(task)

    def _describe_tools(self) -> str:
        """生成可用工具的描述文本，供规划阶段参考。"""
        tools = self.registry.get_openai_definitions()
        if not tools:
            return "无可用工具（纯 LLM 推理）"

        lines = []
        for t in tools:
            fn = t["function"]
            name = fn["name"]
            desc = fn.get("description", "无描述")
            params = fn.get("parameters", {}).get("properties", {})
            param_desc = ""
            if params:
                param_items = [f"{k}({v.get('type', 'str')})" for k, v in params.items()]
                param_desc = f"  参数: {', '.join(param_items)}"
            lines.append(f"  - {name}: {desc}{param_desc}")
        return "\n".join(lines)

    def _parse_plan(self, text: str) -> list[str]:
        """从 LLM 回复中解析步骤列表。

        兼容多种格式:
        - 数字编号: "1. 步骤内容" / "1) 步骤内容" / "1、步骤内容"
        - 破折号:   "- 步骤内容"
        - 步骤前缀: "步骤1: 步骤内容" / "Step 1: 步骤内容"
        """
        import re

        steps = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # 跳过明显的非步骤行（标题、注释、空壳）
            if any(line.startswith(w) for w in
                   ("以下", "计划", "规划", "执行", "首先", "接下来", "最后", "#")):
                if len(line) < 15:  # 短行才跳，避免误伤长内容
                    continue

            # 剥离编号前缀
            stripped = line
            m = re.match(
                r'^(\d+[.\)\、\s]\s*|步骤\s*\d+[：:]\s*|[Ss]tep\s*\d+[：:]\s*)',
                stripped,
            )
            if m:
                stripped = stripped[m.end():]
            elif stripped.startswith(("- ", "* ", "• ")):
                stripped = stripped[2:]

            stripped = stripped.strip()
            if stripped and len(stripped) > 3:
                steps.append(stripped)

        return steps


# ══════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════

_PLAN_SYSTEM_PROMPT = """你是一个任务规划专家。你的职责是分析用户任务，制定清晰、有序的执行计划。

可用的工具:
{tools}

规划原则:
1. 将复杂任务拆解为 3-7 个独立可执行的步骤
2. 每个步骤应明确、具体，方便后续执行者理解
3. 优先考虑工具的能力，充分利用可用工具
4. 步骤之间应有逻辑先后顺序
5. 只输出步骤列表，每行一个步骤，用"数字. "格式

输出格式示例:
1. 查询上海未来三天的天气情况
2. 根据天气推荐适合的户外活动
3. 整理一份三日行程建议"""

_SUMMARIZE_PROMPT = """请基于以下执行结果，对用户的任务给出完整、清晰的最终回答。

用户任务: {task}

执行计划:
{plan}

各步骤执行结果:
{results}

要求:
- 综合所有步骤的结果，给出完整回答
- 如果某个步骤失败，说明原因并尽可能提供替代方案
- 回答应结构化、有条理，避免简单地罗列各步骤结果"""
