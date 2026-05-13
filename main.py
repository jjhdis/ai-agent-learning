"""
AI Agent 入口 —— 交互式对话。

使用前设置环境变量（秘钥你来填）：
    export LLM_API_KEY="sk-your-deepseek-key"

然后运行:
    python main.py
"""

import sys

from config import Config
from agent.llm.client import LLMClient
from agent.tools.registry import ToolRegistry
from agent.tools.weather import WeatherTool
from agent.core.agent import Agent


def build_agent() -> Agent:
    """组装 Agent：配置 LLM + 注册工具。"""
    llm = LLMClient(Config.llm)
    registry = ToolRegistry()
    registry.register(WeatherTool())

    return Agent(
        llm_client=llm,
        registry=registry,
        hooks={
            "on_start": lambda msg: print(f"\n{'='*50}\n[You]: {msg}"),
            "on_tool_call": lambda name, args: print(f"[Tool] 调用工具: {name}({args})"),
            "on_reply": lambda msg: print(f"[AI]: {msg}"),
        },
    )


def demo_travel_chain(agent):
    """演示管道（Chain）的威力：十一假期出行推荐。

    展示如何把多个步骤串成管道，前一步的输出是后一步的输入。
    """
    from agent.chain import RunnablePassthrough, RunnableMap, RunnableLambda
    import json

    # ---- 步骤1：分析用户偏好 ----
    def analyze_preference(user_input: str) -> dict:
        """让 Agent 分析用户的旅行偏好。"""
        prompt = (
            f"用户说：\"{user_input}\"\n\n"
            f"请从以下维度分析用户的旅行偏好，只返回 JSON 格式（不要其他文字）：\n"
            f'- "preference": 用户偏好（"自然风光"/"城市观光"/"海滨度假"/"文化历史"）\n'
            f'- "budget": 预算倾向（"经济"/"中等"/"奢华"）\n'
            f'- "with_who": 和谁去（"独自"/"情侣"/"家庭"/"朋友"）\n'
        )
        result = agent.run(prompt)
        # 从结果中提取 JSON
        try:
            # 尝试直接解析
            return json.loads(result)
        except json.JSONDecodeError:
            # 如果 LLM 返回了额外文字，尝试从中提取 JSON 部分
            import re
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"preference": "自然风光", "budget": "中等", "with_who": "独自"}

    # ---- 步骤2：根据偏好推荐候选城市 ----
    def recommend_cities(prefs: dict) -> dict:
        """根据偏好推荐 3 个候选城市，返回包含偏好和城市的 dict。"""
        prompt = (
            f"用户旅行偏好：\n"
            f"- 偏好类型：{prefs['preference']}\n"
            f"- 预算：{prefs['budget']}\n"
            f"- 和谁去：{prefs['with_who']}\n\n"
            f"请推荐 3 个适合十一假期（10月1日-10月7日）去的中国城市。\n"
            f"只返回 JSON 格式的城市列表，例如：[\"成都\", \"西安\", \"大理\"]"
        )
        result = agent.run(prompt)
        try:
            parsed = json.loads(result)
            # LLM 可能返回 ["成都", "西安", "大理"] 或 {"cities": ["成都", "西安", "大理"]}
            if isinstance(parsed, list):
                cities = parsed
            elif isinstance(parsed, dict):
                # 尝试从 dict 中提取城市列表
                for key in ("cities", "city", "recommendations"):
                    if key in parsed and isinstance(parsed[key], list):
                        cities = parsed[key]
                        break
                else:
                    cities = ["成都", "西安", "大理"]
            else:
                cities = ["成都", "西安", "大理"]
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', result, re.DOTALL)
            cities = json.loads(match.group()) if match else ["成都", "西安", "大理"]

        # 把偏好和城市合并到一个 dict，方便后续步骤使用
        return {"preferences": prefs, "cities": cities}

    # ---- 步骤3：并行查多个城市的天气 ----
    def check_cities_weather(data: dict) -> dict:
        """用 RunnableMap 并行查询多个城市的天气。

        由于现在是 5 月，查不到十一假期的真实天气，
        所以用模拟数据来演示管道的并行查询能力。
        """
        cities = data["cities"]

        # 模拟的十一假期天气数据（10月1日-10月7日）
        mock_weather = {
            "稻城亚丁": (
                "稻城亚丁 10月天气：秋季晴朗为主，昼夜温差极大。"
                "白天 10~15°C，夜间可降至 0°C以下。"
                "建议带羽绒服、保暖衣物，注意高反。"
            ),
            "桂林": (
                "桂林 10月天气：秋季温和舒适，偶有阵雨。"
                "白天 25~30°C，夜间 18~22°C。"
                "建议带薄外套和雨具，适合户外活动。"
            ),
            "张家界": (
                "张家界 10月天气：秋季凉爽，可能有小雨。"
                "白天 20~28°C，夜间 15~18°C。"
                "建议带外套和雨具，云雾缭绕景色美。"
            ),
            "成都": (
                "成都 10月天气：秋季凉爽，多云为主。"
                "白天 20~25°C，夜间 15~18°C。"
                "适合吃火锅、逛宽窄巷子，建议带薄外套。"
            ),
            "西安": (
                "西安 10月天气：秋季晴朗，气温适宜。"
                "白天 18~24°C，夜间 10~15°C。"
                "适合游览兵马俑、古城墙，建议带外套。"
            ),
            "大理": (
                "大理 10月天气：秋季晴好，阳光充足。"
                "白天 20~26°C，夜间 10~15°C。"
                "适合环洱海骑行，建议带防晒和薄外套。"
            ),
            "北京": (
                "北京 10月天气：秋高气爽，非常适合旅游。"
                "白天 18~24°C，夜间 8~12°C。"
                "建议带外套，香山红叶正值最佳观赏期。"
            ),
            "杭州": (
                "杭州 10月天气：秋季舒适，桂花飘香。"
                "白天 22~28°C，夜间 15~19°C。"
                "适合游西湖、品龙井，建议带薄外套。"
            ),
            "青岛": (
                "青岛 10月天气：秋季凉爽，海风较大。"
                "白天 18~23°C，夜间 12~16°C。"
                "适合海边漫步、吃海鲜，建议带防风外套。"
            ),
        }

        # 为每个城市创建一个查天气的 Runnable（使用模拟数据）
        # 用函数工厂避免闭包陷阱
        def make_weather_runnable(city_name):
            return RunnableLambda(lambda _: mock_weather.get(city_name, f"{city_name} 10月天气：气温适宜，适合旅游。"))

        weather_checks = {city: make_weather_runnable(city) for city in cities}

        # 并行执行所有天气查询
        weathers = RunnableMap(weather_checks).invoke(None)
        data["weathers"] = weathers
        return data

    # ---- 步骤4：综合给出推荐 ----
    def make_recommendation(data: dict) -> str:
        """综合偏好和天气，给出最终推荐。"""
        prefs = data["preferences"]
        cities = data["cities"]
        weathers = data["weathers"]

        weather_report = "\n".join([f"{city}: {weathers[city][:100]}" for city in cities])

        prompt = (
            f"用户偏好：{prefs}\n"
            f"候选城市：{cities}\n"
            f"各城市天气情况：\n{weather_report}\n\n"
            f"请综合天气和用户偏好，给出十一假期的出行推荐。\n"
            f"要求：\n"
            f"1. 推荐最合适的 1-2 个城市\n"
            f"2. 说明推荐理由（结合天气和偏好）\n"
            f"3. 给出简单的出行建议（带什么衣服、注意什么）\n"
        )
        return agent.run(prompt)

    print("\n" + "=" * 60)
    print("十一假期出行推荐 —— Chain 管道演示")
    print("=" * 60)
    print("\n【管道结构】")
    print("  步骤1: 分析偏好  →  步骤2: 推荐城市  →  步骤3: 查天气  →  步骤4: 综合推荐")
    print("  (RunnableLambda)  (RunnableLambda)   (RunnableMap)    (RunnableLambda)")
    print("\n" + "-" * 60)

    # 组装管道：前一步的输出自动成为后一步的输入
    travel_chain = (
        RunnableLambda(analyze_preference)      # 步骤1：分析偏好 → 输出 dict
        | RunnableLambda(recommend_cities)       # 步骤2：推荐城市 → 输出 dict（含偏好+城市）
        | RunnableLambda(check_cities_weather)   # 步骤3：查天气（内部并行）→ 输出 dict（含天气）
        | RunnableLambda(make_recommendation)    # 步骤4：综合推荐 → 输出最终文本
    )

    user_input = "十一假期我想出去玩，有什么推荐？"
    print(f"\n[用户输入]: {user_input}")
    print("-" * 60)

    result = travel_chain.invoke(user_input)

    print("\n" + "=" * 60)
    print("[最终推荐]")
    print(result)
    print("=" * 60)


def main():
    print("=" * 50)
    print(f"[AI Agent 已启动]")
    print(f"   模型: {Config.llm.model}")
    print(f"   工具: 天气查询 (get_weather)")
    print(f"   输入 '/quit' 退出, '/reset' 清空历史, '/demo' 运行 Chain 演示")
    print("=" * 50)

    agent = build_agent()

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
        if user_input.lower() == "/demo":
            demo_travel_chain(agent)
            continue
        if user_input.lower() == "/reset":
            agent.reset()
            print("[对话历史已清空]")
            continue

        try:
            agent.run(user_input)
        except Exception as e:
            print(f"[错误] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
