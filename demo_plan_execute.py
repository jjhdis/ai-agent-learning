"""
Plan-and-Execute Agent 演示 —— 先规划再执行的两阶段 Agent。

演示场景:
    1. 旅行规划（含工具调用）—— 查询天气 + 检索知识 + 制定行程
    2. 对比 ReAct Agent 的差异

运行:
    python demo_plan_execute.py

环境变量（可选，有默认值）:
    LLM_API_KEY  - 模型 API 密钥
    LLM_BASE_URL - 模型 API 地址
    LLM_MODEL    - 模型名称
"""

import sys
import io

# Windows GBK 控制台无法输出 emoji，强制使用 UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import Config
from agent.llm.client import LLMClient
from agent.tools.registry import ToolRegistry
from agent.tools.weather import WeatherTool
from agent.plan_execute import PlanAndExecuteAgent
from agent.core.agent import Agent


def setup():
    """组装 LLM 客户端和工具注册中心。"""
    llm = LLMClient(Config.llm)
    registry = ToolRegistry()
    registry.register(WeatherTool())
    return llm, registry


def build_knowledge_tool():
    """构建 RAG 知识检索工具，加载旅行相关知识文档。

    如果 Embedding API 不可用（网络问题等），返回 None 并降级为无 RAG 的演示。
    """
    from agent.rag.retriever import RetrieverTool

    travel_texts = [
        (
            "上海三日游经典路线: "
            "Day1 外滩→南京路步行街→人民广场→上海博物馆→豫园城隍庙，晚上看外滩夜景；"
            "Day2 迪士尼乐园全天（建议工作日去，人少排队短）；"
            "Day3 田子坊→新天地→武康路→思南路，感受老上海风情。"
            "交通建议: 地铁出行最方便，买一日票或三日票更划算。"
        ),
        (
            "上海雨天备选方案: "
            "室内景点推荐 —— 上海科技馆（浦东，适合亲子）、上海自然博物馆（静安，需预约）、"
            "上海海洋水族馆（陆家嘴，亚洲最大之一）、上海电影博物馆（徐汇）、"
            "1933老场坊（虹口，创意园区，适合拍照）。"
            "商场推荐 —— 环球港（普陀，超大）、国金中心（陆家嘴，高端）、"
            "来福士（人民广场，方便）。"
        ),
        (
            "上海特色美食推荐: "
            "早餐——小笼包（南翔馒头店、富春小笼）、生煎（小杨生煎、大壶春）、"
            "粢饭糕、咸豆浆；"
            "午餐——本帮菜（老正兴、上海老饭店）、葱油拌面、蟹粉面（方亮）；"
            "晚餐——红烧肉、油爆虾、腌笃鲜、八宝鸭。"
            "小吃街——云南南路美食街、吴江路小吃街、城隍庙小吃广场。"
            "预算参考: 简餐人均30-50元，正餐人均80-150元，高端餐厅300+元。"
        ),
        (
            "上海住宿建议: "
            "经济型——如家、汉庭等连锁酒店，价格200-400元/晚，建议选地铁站附近；"
            "舒适型——全季、亚朵等中档酒店，400-700元/晚，服务好环境佳；"
            "高端型——和平饭店、半岛酒店等，1500元+/晚，外滩周边。"
            "推荐区域: 人民广场/南京东路（市中心出行方便）、静安寺（安静有格调）、"
            "陆家嘴（商务区，夜景好）。旅游旺季（五一、十一、春节）需提前2-4周预订。"
        ),
    ]
    travel_metadatas = [
        {"topic": "上海三日游", "category": "经典路线"},
        {"topic": "上海雨天", "category": "室内活动"},
        {"topic": "上海美食", "category": "餐饮"},
        {"topic": "上海住宿", "category": "住宿"},
    ]

    try:
        tool = RetrieverTool(k=3)
        tool.add_texts(travel_texts, travel_metadatas)
    except Exception as e:
        print(f"  [WARN] Embedding API 不可用 ({e})，降级为无 RAG 演示")
        return None

    return tool


# ══════════════════════════════════════════════════════════════════════
# Demo 1: 旅行规划（Plan-and-Execute）
# ══════════════════════════════════════════════════════════════════════

def demo_travel_planning():
    """演示 Plan-and-Execute Agent 处理旅行规划任务。

    这是一个典型的多步任务: 需要查天气→检索攻略→制定行程，
    Plan-and-Execute 模式天然适合这种需要先整体规划再逐步执行的场景。
    """
    print("\n" + "=" * 60)
    print("Demo 1: Plan-and-Execute 旅行规划")
    print("=" * 60)

    llm, registry = setup()

    # 添加知识检索工具（Embedding API 不可用时降级跳过）
    retriever_tool = build_knowledge_tool()
    if retriever_tool:
        registry.register(retriever_tool)

    print(f"\n已注册工具: {[t['function']['name'] for t in registry.get_openai_definitions()]}")

    agent = PlanAndExecuteAgent(llm, registry, verbose=True)

    task = (
        "帮我规划一次上海三日游。需要考虑: "
        "1) 查询上海未来几天的天气，根据天气推荐室内或室外活动；"
        "2) 检索上海旅游攻略和美食推荐；"
        "3) 根据以上信息制定一份完整的三日行程安排；"
        "4) 给出住宿建议和预算预估。"
    )

    print(f"\n[任务] {task}\n")
    result = agent.run(task)

    print(f"\n{'='*60}")
    print("最终回答:")
    print(f"{'='*60}")
    print(result)


# ══════════════════════════════════════════════════════════════════════
# Demo 2: 对比 ReAct Agent
# ══════════════════════════════════════════════════════════════════════

def demo_react_comparison():
    """对比 Plan-and-Execute 和 ReAct Agent 在复杂任务上的表现差异。

    ReAct 是边走边看（思考→行动→观察循环），适合即时反馈的任务；
    Plan-and-Execute 是先规划再执行，适合需要全局视野的复杂任务。
    """
    print("\n" + "=" * 60)
    print("Demo 2: ReAct Agent 对比演示")
    print("=" * 60)

    llm, registry = setup()

    print("\n[说明] 同样的任务，ReAct Agent 会即时反应，而 Plan-and-Execute 会先规划。")
    print("对比体现了两者的设计哲学差异。\n")

    react_agent = Agent(llm_client=llm, registry=registry)

    task = "帮我分析一下：上海明天天气怎么样？如果下雨有哪些室内景点可以去？推荐一个雨天一日游方案。"

    print(f"[任务] {task}\n")
    print(">>> ReAct Agent 执行:")
    result = react_agent.run(task)
    print(f"\n[ReAct 回复]\n{result}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    """运行所有演示。"""
    print("=" * 60)
    print("Plan-and-Execute Agent 演示")
    print("=" * 60)
    print()
    print("Plan-and-Execute vs ReAct:")
    print("  ReAct:  思考 → 行动 → 观察 → 思考 → ...（交替进行，边走边看）")
    print("  P&E:    制定完整计划 → 逐步执行 → 汇总结果（先规划后执行）")
    print()
    print("适用场景:")
    print("  P&E 适合: 复杂多步任务、需要全局视野、可分解为独立子任务")
    print("  ReAct 适合: 简单问答、需要即时反馈、单步即可完成")
    print()

    # Demo 1: Plan-and-Execute 核心演示
    demo_travel_planning()

    # Demo 2: 对比 ReAct
    print("\n")
    demo_react_comparison()

    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
