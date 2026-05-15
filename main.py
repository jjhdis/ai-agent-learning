"""
AI Agent 入口 —— 交互式对话。

使用前设置环境变量（秘钥你来填）：
    export LLM_API_KEY="sk-your-deepseek-key"

然后运行:
    python main.py

命令:
    /quit           退出
    /reset          清空对话历史与记忆
    /memory buffer  切换到完整记忆模式（默认）
    /memory window  切换到滑动窗口记忆模式
    /memory summary 切换到摘要记忆模式
    /memory off     关闭记忆
    /demo-memory    运行记忆系统演示
    /demo-prompt    运行提示词模板系统演示
"""

import sys

from config import Config
from agent.llm.client import LLMClient
from agent.tools.registry import ToolRegistry
from agent.tools.weather import WeatherTool
from agent.core.agent import Agent
from agent.memory.buffer import ConversationBufferMemory
from agent.memory.window import ConversationBufferWindowMemory
from agent.memory.summary import ConversationSummaryMemory
from agent.callback.manager import CallbackManager
from agent.callback.logging import LoggingCallback
from agent.callback.token_counting import TokenCountingCallback
from agent.prompt.base import PromptTemplate
from agent.prompt.chat import ChatPromptTemplate
from agent.prompt.placeholder import MessagePlaceholder
from agent.prompt.few_shot import (
    LengthBasedExampleSelector,
    FewShotPromptTemplate,
)


# ──────────────────────────────────────────────
# 组装
# ──────────────────────────────────────────────

def build_agent(memory=None, callbacks=None):
    """组装 Agent：配置 LLM + 注册工具 + 可选记忆 + 可选回调。"""
    llm = LLMClient(Config.llm)
    registry = ToolRegistry()
    registry.register(WeatherTool())

    return Agent(
        llm_client=llm,
        registry=registry,
        memory=memory,
        hooks={
            "on_start": lambda msg: print(f"\n{'='*50}\n[You]: {msg}"),
            "on_tool_call": lambda name, args: print(f"[Tool] 调用工具: {name}({args})"),
            "on_reply": lambda msg: print(f"[AI]: {msg}"),
        },
        callbacks=callbacks,
    )


# ──────────────────────────────────────────────
# 记忆演示
# ──────────────────────────────────────────────

def demo_memory(agent_base):
    """演示三种记忆模式：连续两轮对话验证记忆效果。"""
    print("\n" + "=" * 60)
    print("记忆系统演示 —— 跨轮次上下文记忆")
    print("=" * 60)

    user_messages = [
        "上海今天天气怎么样？",
        "北京呢？",  # 依赖上一轮的上下文
    ]

    for mem_type, mem_instance in [
        ("完整记忆 (Buffer)", ConversationBufferMemory()),
        ("滑动窗口记忆 (Window, K=2)", ConversationBufferWindowMemory(k=2)),
        ("摘要记忆 (Summary)", ConversationSummaryMemory(llm=agent_base.llm, buffer_size=4)),
    ]:
        print(f"\n{'─'*50}")
        print(f"[模式] {mem_type}")
        print(f"{'─'*50}")

        agent = build_agent(memory=mem_instance)
        for msg in user_messages:
            print(f"\n[You]: {msg}")
            agent.run(msg)

        agent.reset()


# ──────────────────────────────────────────────
# 回调系统演示
# ──────────────────────────────────────────────

def demo_callback(agent_base):
    """演示回调系统：展示 LoggingCallback 记录的所有生命周期事件。"""
    print("\n" + "=" * 60)
    print("回调系统演示 —— 生命周期事件追踪")
    print("=" * 60)

    callbacks = CallbackManager()
    callbacks.add_handler(LoggingCallback(verbose=True))

    agent = build_agent(memory=None, callbacks=callbacks)
    print("\n[说明] LoggingCallback 将记录以下事件:")
    print("  on_agent_start → on_llm_start → on_llm_end → on_think")
    print("  → on_tool_start → on_tool_end → on_llm_start → on_llm_end")
    print("  → on_agent_end（含耗时统计）")
    print()

    agent.run("上海今天天气怎么样？")

    print("\n[统计]")
    print(f"  LLM 调用次数: {callbacks.handlers[0].llm_call_count}")
    print(f"  工具调用次数: {callbacks.handlers[0].tool_call_count}")
    print(f"  总耗时: {callbacks.handlers[0].elapsed:.2f}s")


# ──────────────────────────────────────────────
# Token 计数演示
# ──────────────────────────────────────────────

def demo_token():
    """演示 TokenCountingCallback：展示每次 LLM 调用的 Token 消耗和预估费用。"""
    print("\n" + "=" * 60)
    print("Token 计数演示 —— 统计每次 LLM 调用的 Token 消耗")
    print("=" * 60)

    callbacks = CallbackManager()
    callbacks.add_handler(TokenCountingCallback(verbose=True))

    agent = build_agent(memory=None, callbacks=callbacks)
    print("\n[说明] TokenCountingCallback 将记录每次 LLM 调用的 Token 数:")
    print("  on_llm_end → 提取 usage.prompt_tokens / completion_tokens / total_tokens")
    print("  on_agent_end → 汇总统计 + 预估费用")
    print()

    agent.run("上海今天天气怎么样？")

    counter = callbacks.handlers[0]
    print("\n[详细统计]")
    print(f"  模型: {counter._actual_model}")
    print(f"  LLM 调用次数: {counter.llm_call_count}")
    print(f"  输入 Token:   {counter.prompt_tokens:,}")
    print(f"  输出 Token:   {counter.completion_tokens:,}")
    print(f"  总 Token:     {counter.total_tokens:,}")
    print(f"  预估费用:     ¥{counter.estimated_cost:.6f}")


# ──────────────────────────────────────────────
# 提示词模板演示
# ──────────────────────────────────────────────

def demo_prompt():
    """演示提示词模板系统：PromptTemplate / ChatPromptTemplate / Few-Shot / MessagePlaceholder。"""
    print("\n" + "=" * 60)
    print("提示词模板系统演示 —— Prompt Template")
    print("=" * 60)

    # ── 1. PromptTemplate 基础 ──
    print("\n" + "─" * 50)
    print("1. PromptTemplate —— 字符串模板变量替换")
    print("─" * 50)

    tpl = PromptTemplate("你好，{name}！今天是{date}，{city}天气怎么样？")
    result = tpl.format(name="小明", date="2026-05-15", city="上海")
    print(f"   模板: {tpl!r}")
    print(f"   变量: {tpl.input_variables}")
    print(f"   结果: {result}")

    # ── 2. partial() 部分变量绑定 ──
    print("\n" + "─" * 50)
    print("2. partial() —— 预设部分变量，延迟填充")
    print("─" * 50)

    base_tpl = PromptTemplate("城市: {city}, 日期: {date}, 用户: {user}")
    pre_tpl = base_tpl.partial(date="2026-05-15", user="测试员")
    print(f"   基础模板: {base_tpl.input_variables}")
    print(f"   预设变量后: 剩余变量 = {pre_tpl.input_variables}")
    print(f"   最终填充: {pre_tpl.format(city='北京')}")

    # ── 3. PromptTemplate 作为 Runnable ──
    print("\n" + "─" * 50)
    print("3. Runnable 协议 —— 模板作为管道组件")
    print("─" * 50)

    from agent.chain.passthrough import RunnableLambda

    tpl = PromptTemplate("分析以下内容:\n{input}\n\n请给出摘要。")
    pipeline = RunnableLambda(lambda x: x.strip()) | tpl
    final = pipeline.invoke("  这是一段需要分析的文本。  ")
    print(f"   管道: StripText | PromptTemplate")
    print(f"   输入: '  这是一段需要分析的文本。  '")
    print(f"   输出: {final[:60]}...")

    # ── 4. ChatPromptTemplate ──
    print("\n" + "─" * 50)
    print("4. ChatPromptTemplate —— 多角色对话模板")
    print("─" * 50)

    chat_tpl = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}助手，擅长{skill}。回答风格：{style}。"),
        ("user", "{input}"),
    ])
    msgs = chat_tpl.format_messages(
        role="天气",
        skill="气象预报",
        style="简洁明了",
        input="上海今天天气怎么样？",
    )
    print(f"   模板消息数: {len(chat_tpl._messages)}")
    print(f"   生成的对话消息:")
    for m in msgs:
        role = m["role"]
        content_preview = m["content"][:80]
        print(f"      [{role}]: {content_preview}")

    # ── 5. MessagePlaceholder ──
    print("\n" + "─" * 50)
    print("5. MessagePlaceholder —— 运行时插入对话历史")
    print("─" * 50)

    history_placeholder = MessagePlaceholder("history")
    tpl_with_history = ChatPromptTemplate([
        ("system", "你是一个{role}助手"),
        history_placeholder,
        ("user", "{input}"),
    ])

    fake_history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "你想知道哪个城市的天气？"},
    ]

    msgs = tpl_with_history.format_messages(
        role="天气",
        input="上海呢？",
        history=fake_history,
    )
    print(f"   占位符: {history_placeholder!r}")
    print(f"   生成的消息列表 (共 {len(msgs)} 条):")
    for i, m in enumerate(msgs):
        role = m["role"]
        content_preview = m["content"][:60]
        print(f"      [{i}] {role}: {content_preview}")

    # ── 6. Few-Shot 示例选择 ──
    print("\n" + "─" * 50)
    print("6. FewShotPromptTemplate —— 动态示例选择")
    print("─" * 50)

    selector = LengthBasedExampleSelector(
        examples=[
            {"input": "北京天气怎么样？", "answer": "北京今天晴，25°C，建议穿薄外套。"},
            {"input": "上海天气怎么样？", "answer": "上海今天多云，28°C，湿度较大。"},
            {"input": "广州天气怎么样？", "answer": "广州今天有雨，30°C，记得带伞。"},
            {"input": "哈尔滨天气怎么样？", "answer": "哈尔滨下雪，-5°C，注意保暖！"},
            {"input": "三亚天气怎么样？", "answer": "三亚晴，32°C，适合游泳和潜水。"},
        ],
        example_prompt=PromptTemplate("问题: {input}\n回答: {answer}"),
        max_tokens=200,
    )

    few_shot = FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=PromptTemplate("问题: {input}\n回答: {answer}"),
        prefix="以下是一些天气查询的示例:\n",
        suffix="\n\n现在请回答: {input}",
        example_separator="\n\n",
    )

    # 模拟对 "南方城市" 的查询，选择器会优先选取最近的示例
    prompt = few_shot.format(input="深圳天气怎么样？")
    selected_count = len(selector.select_examples({"input": "深圳天气怎么样？"}))
    print(f"   示例池: {len(selector._examples)} 个")
    print(f"   本次选中: {selected_count} 个 (max_tokens={selector.max_tokens})")
    print(f"   生成的提示词 ({len(prompt)} 字符):")
    print(f"   {'─'*40}")
    for line in prompt.split("\n"):
        print(f"   | {line}")
    print(f"   {'─'*40}")

    # 演示添加新示例
    selector.add_example({"input": "深圳天气怎么样？", "answer": "深圳晴，29°C，建议防晒。"})
    print(f"\n   添加新示例后，示例池: {len(selector._examples)} 个")

    # ── 7. 综合演示：模板 + Agent ──
    print("\n" + "─" * 50)
    print("7. 综合演示 —— 用模板构建 Agent 的 System Prompt")
    print("─" * 50)

    system_tpl = PromptTemplate(
        "你是一个{role}助手。你的说话风格是{style}。"
        "当前日期是{date}。你的名字叫{name}。"
    )
    system_msg = system_tpl.format(
        role="天气预报",
        style="活泼风趣",
        date="2026-05-15",
        name="小天气",
    )
    print(f"   生成的 System Prompt: {system_msg}")

    # ── 8. 模板异常处理 ──
    print("\n" + "─" * 50)
    print("8. 异常处理 —— 缺少变量时的友好提示")
    print("─" * 50)

    try:
        PromptTemplate("你好，{name}！今天{city}天气怎么样？").format(name="小明")
    except KeyError as e:
        print(f"   缺少变量时抛出: {e}")

    # 使用 partial 避免异常
    safe_tpl = PromptTemplate("你好，{name}！今天{city}天气怎么样？").partial(city="默认城市")
    print(f"   使用 partial 兜底后: {safe_tpl.format(name='小明')}")

    print("\n" + "=" * 60)
    print("提示词模板系统演示完成！")
    print("=" * 60)

def main():
    memory = ConversationBufferMemory()
    memory_label = "完整记忆 (Buffer)"

    print("=" * 50)
    print(f"[AI Agent 已启动]")
    print(f"   模型: {Config.llm.model}")
    print(f"   工具: 天气查询 (get_weather)")
    print(f"   记忆: {memory_label}")
    print(f"   输入 /quit 退出, /reset 清空, /demo-memory 演示记忆, /demo-callback 演示回调, /demo-token 演示Token计数, /demo-prompt 演示模板, /memory 切换记忆模式")
    print("=" * 50)

    agent = build_agent(memory=memory)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[再见!]")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("[再见!]")
            break
        if user_input.lower() == "/reset":
            agent.reset()
            print("[对话历史已清空]")
            continue
        if user_input.lower() == "/demo-memory":
            demo_memory(agent)
            continue
        if user_input.lower() == "/demo-callback":
            demo_callback(agent)
            continue
        if user_input.lower() == "/demo-token":
            demo_token()
            continue
        if user_input.lower() == "/demo-prompt":
            demo_prompt()
            continue
        if user_input.lower().startswith("/memory"):
            parts = user_input.split(maxsplit=1)
            mode = parts[1].strip().lower() if len(parts) > 1 else ""

            if mode == "buffer":
                memory = ConversationBufferMemory()
                memory_label = "完整记忆 (Buffer)"
            elif mode == "window":
                memory = ConversationBufferWindowMemory(k=3)
                memory_label = "滑动窗口记忆 (Window, K=3)"
            elif mode == "summary":
                memory = ConversationSummaryMemory(llm=agent.llm, buffer_size=4)
                memory_label = "摘要记忆 (Summary)"
            elif mode == "off":
                memory = None
                memory_label = "无"
            else:
                print(f"[用法] /memory buffer|window|summary|off  当前: {memory_label}")
                continue

            agent = build_agent(memory=memory)
            print(f"[记忆模式] 已切换为: {memory_label}")
            continue

        try:
            agent.run(user_input)
        except Exception as e:
            print(f"[错误] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════
# 以下为第一阶段（Chain 管道）学习代码，已完成学习目标，暂注释保留。
# ══════════════════════════════════════════════════════════════════════

# def demo_travel_chain(agent):
#     """演示管道（Chain）的威力：十一假期出行推荐。
#
#     展示如何把多个步骤串成管道，前一步的输出是后一步的输入。
#     """
#     from agent.chain import RunnablePassthrough, RunnableMap, RunnableLambda
#     import json
#
#     # ---- 步骤1：分析用户偏好 ----
#     def analyze_preference(user_input: str) -> dict:
#         """让 Agent 分析用户的旅行偏好。"""
#         prompt = (
#             f"用户说：\"{user_input}\"\n\n"
#             f"请从以下维度分析用户的旅行偏好，只返回 JSON 格式（不要其他文字）：\n"
#             f'- "preference": 用户偏好（"自然风光"/"城市观光"/"海滨度假"/"文化历史"）\n'
#             f'- "budget": 预算倾向（"经济"/"中等"/"奢华"）\n'
#             f'- "with_who": 和谁去（"独自"/"情侣"/"家庭"/"朋友"）\n'
#         )
#         result = agent.run(prompt)
#         try:
#             return json.loads(result)
#         except json.JSONDecodeError:
#             import re
#             match = re.search(r'\{.*\}', result, re.DOTALL)
#             if match:
#                 return json.loads(match.group())
#             return {"preference": "自然风光", "budget": "中等", "with_who": "独自"}
#
#     # ---- 步骤2：根据偏好推荐候选城市 ----
#     def recommend_cities(prefs: dict) -> dict:
#         """根据偏好推荐 3 个候选城市，返回包含偏好和城市的 dict。"""
#         prompt = (
#             f"用户旅行偏好：\n"
#             f"- 偏好类型：{prefs['preference']}\n"
#             f"- 预算：{prefs['budget']}\n"
#             f"- 和谁去：{prefs['with_who']}\n\n"
#             f"请推荐 3 个适合十一假期（10月1日-10月7日）去的中国城市。\n"
#             f"只返回 JSON 格式的城市列表，例如：[\"成都\", \"西安\", \"大理\"]"
#         )
#         result = agent.run(prompt)
#         try:
#             parsed = json.loads(result)
#             if isinstance(parsed, list):
#                 cities = parsed
#             elif isinstance(parsed, dict):
#                 for key in ("cities", "city", "recommendations"):
#                     if key in parsed and isinstance(parsed[key], list):
#                         cities = parsed[key]
#                         break
#                 else:
#                     cities = ["成都", "西安", "大理"]
#             else:
#                 cities = ["成都", "西安", "大理"]
#         except json.JSONDecodeError:
#             import re
#             match = re.search(r'\[.*\]', result, re.DOTALL)
#             cities = json.loads(match.group()) if match else ["成都", "西安", "大理"]
#
#         return {"preferences": prefs, "cities": cities}
#
#     # ---- 步骤3：并行查多个城市的天气 ----
#     def check_cities_weather(data: dict) -> dict:
#         """用 RunnableMap 并行查询多个城市的天气。"""
#         cities = data["cities"]
#
#         mock_weather = {
#             "稻城亚丁": (
#                 "稻城亚丁 10月天气：秋季晴朗为主，昼夜温差极大。"
#                 "白天 10~15°C，夜间可降至 0°C以下。"
#                 "建议带羽绒服、保暖衣物，注意高反。"
#             ),
#             "桂林": (
#                 "桂林 10月天气：秋季温和舒适，偶有阵雨。"
#                 "白天 25~30°C，夜间 18~22°C。"
#                 "建议带薄外套和雨具，适合户外活动。"
#             ),
#             "张家界": (
#                 "张家界 10月天气：秋季凉爽，可能有小雨。"
#                 "白天 20~28°C，夜间 15~18°C。"
#                 "建议带外套和雨具，云雾缭绕景色美。"
#             ),
#             "成都": (
#                 "成都 10月天气：秋季凉爽，多云为主。"
#                 "白天 20~25°C，夜间 15~18°C。"
#                 "适合吃火锅、逛宽窄巷子，建议带薄外套。"
#             ),
#             "西安": (
#                 "西安 10月天气：秋季晴朗，气温适宜。"
#                 "白天 18~24°C，夜间 10~15°C。"
#                 "适合游览兵马俑、古城墙，建议带外套。"
#             ),
#             "大理": (
#                 "大理 10月天气：秋季晴好，阳光充足。"
#                 "白天 20~26°C，夜间 10~15°C。"
#                 "适合环洱海骑行，建议带防晒和薄外套。"
#             ),
#             "北京": (
#                 "北京 10月天气：秋高气爽，非常适合旅游。"
#                 "白天 18~24°C，夜间 8~12°C。"
#                 "建议带外套，香山红叶正值最佳观赏期。"
#             ),
#             "杭州": (
#                 "杭州 10月天气：秋季舒适，桂花飘香。"
#                 "白天 22~28°C，夜间 15~19°C。"
#                 "适合游西湖、品龙井，建议带薄外套。"
#             ),
#             "青岛": (
#                 "青岛 10月天气：秋季凉爽，海风较大。"
#                 "白天 18~23°C，夜间 12~16°C。"
#                 "适合海边漫步、吃海鲜，建议带防风外套。"
#             ),
#         }
#
#         def make_weather_runnable(city_name):
#             return RunnableLambda(lambda _: mock_weather.get(city_name, f"{city_name} 10月天气：气温适宜，适合旅游。"))
#
#         weather_checks = {city: make_weather_runnable(city) for city in cities}
#         weathers = RunnableMap(weather_checks).invoke(None)
#         data["weathers"] = weathers
#         return data
#
#     # ---- 步骤4：综合给出推荐 ----
#     def make_recommendation(data: dict) -> str:
#         """综合偏好和天气，给出最终推荐。"""
#         prefs = data["preferences"]
#         cities = data["cities"]
#         weathers = data["weathers"]
#
#         weather_report = "\n".join([f"{city}: {weathers[city][:100]}" for city in cities])
#
#         prompt = (
#             f"用户偏好：{prefs}\n"
#             f"候选城市：{cities}\n"
#             f"各城市天气情况：\n{weather_report}\n\n"
#             f"请综合天气和用户偏好，给出十一假期的出行推荐。\n"
#             f"要求：\n"
#             f"1. 推荐最合适的 1-2 个城市\n"
#             f"2. 说明推荐理由（结合天气和偏好）\n"
#             f"3. 给出简单的出行建议（带什么衣服、注意什么）\n"
#         )
#         return agent.run(prompt)
#
#     print("\n" + "=" * 60)
#     print("十一假期出行推荐 —— Chain 管道演示")
#     print("=" * 60)
#     print("\n【管道结构】")
#     print("  步骤1: 分析偏好  →  步骤2: 推荐城市  →  步骤3: 查天气  →  步骤4: 综合推荐")
#     print("  (RunnableLambda)  (RunnableLambda)   (RunnableMap)    (RunnableLambda)")
#     print("\n" + "-" * 60)
#
#     travel_chain = (
#         RunnableLambda(analyze_preference)
#         | RunnableLambda(recommend_cities)
#         | RunnableLambda(check_cities_weather)
#         | RunnableLambda(make_recommendation)
#     )
#
#     user_input = "十一假期我想出去玩，有什么推荐？"
#     print(f"\n[用户输入]: {user_input}")
#     print("-" * 60)
#
#     result = travel_chain.invoke(user_input)
#
#     print("\n" + "=" * 60)
#     print("[最终推荐]")
#     print(result)
#     print("=" * 60)
