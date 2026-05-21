"""
第 8b 阶段演示 —— Self-Ask Agent + XML Agent + Conversational Agent。

展示三种不同的 Agent 类型及其工具调用方式:
    1. Self-Ask Agent  —— 纯文本 "Follow up:" / "So the final answer is:" 模式
    2. XML Agent       —— XML 标签 <tool_call> / <tool_result> 模式
    3. Conversational  —— 现有 ReAct Agent + Memory 就是 Conversational Agent

运行:
    python demo_phase8b.py
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
from agent.core.agent import Agent
from agent.memory.buffer import ConversationBufferMemory
from agent.self_ask import SelfAskAgent
from agent.xml_agent import XMLAgent


# ══════════════════════════════════════════════════════════════════════
# 静态知识库 —— 所有 Demo 共用的搜索后端
# ══════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = {
    "诺贝尔物理学奖 2024": (
        "2024年诺贝尔物理学奖授予了John Hopfield（约翰·霍普菲尔德）和"
        "Geoffrey Hinton（杰弗里·辛顿），以表彰他们在人工神经网络和机器学习"
        "方面的基础性发现和发明。他们的工作为现代深度学习和人工智能的蓬勃发展"
        "奠定了关键基础。"
    ),
    "Geoffrey Hinton 贡献": (
        "Geoffrey Hinton被誉为'深度学习之父'。他的主要贡献包括："
        "1) 反向传播算法的推广和应用（1986年）；"
        "2) 玻尔兹曼机（Boltzmann Machine）的发明；"
        "3) 深度信念网络（Deep Belief Networks, 2006年），开创了深度学习新纪元；"
        "4) Dropout正则化技术，防止神经网络过拟合；"
        "5) 2012年ImageNet竞赛中带领团队使用AlexNet取得突破性成果，"
        "证明了深度学习在大规模图像识别中的巨大潜力。"
    ),
    "John Hopfield 贡献": (
        "John Hopfield在1982年提出了Hopfield网络，这是一种递归神经网络模型。"
        "Hopfield网络模拟了人脑的联想记忆机制，能够从不完整或有噪声的输入中"
        "恢复出完整的存储模式。该模型对理解大脑的记忆机制和计算神经科学产生了"
        "深远影响，也为后来的优化算法（如求解旅行商问题）提供了新思路。"
        "Hopfield网络的核心概念——能量函数和吸引子状态，已被广泛应用于各种"
        "机器学习模型中。"
    ),
    "深度学习 对AI的影响": (
        "深度学习自2006年复兴以来，彻底改变了人工智能领域："
        "1) 计算机视觉：在ImageNet图像识别、目标检测、人脸识别等方面超越人类水平；"
        "2) 自然语言处理：Transformer架构和GPT/BERT等大语言模型实现了高质量翻译、"
        "对话、文本生成，直接催生了ChatGPT等现象级应用；"
        "3) 语音识别与合成：准确率大幅提升，已广泛应用于智能助手；"
        "4) 自动驾驶：感知和决策系统取得质的飞跃；"
        "5) 医疗诊断：在医学影像分析、疾病预测等方面达到或超过专家水平；"
        "6) AlphaFold等模型解决了困扰生物学50年的蛋白质折叠问题。"
        "可以说，深度学习是过去十年AI领域最重要的技术突破。"
    ),
    "Python 语言历史": (
        "Python由Guido van Rossum于1991年首次发布。它的设计哲学强调代码可读性和"
        "简洁的语法，使用缩进来划分代码块。Python是解释型、动态类型语言，支持"
        "面向对象、函数式和过程式编程。目前Python是最流行的编程语言之一，尤其在"
        "数据科学、机器学习和Web开发领域占据主导地位。"
    ),
}


def static_search(query: str) -> str:
    """在静态知识库中搜索，返回匹配度最高的结果。

    使用简单的关键词匹配（jaccard-like 词重叠度），
    不需要外部 API，也不需要 Embedding。
    """
    query_words = set(query.lower().split())
    best_score = 0
    best_key = None
    best_content = ""

    for key, content in KNOWLEDGE_BASE.items():
        key_words = set(key.lower().split())
        overlap = len(query_words & key_words)
        # 优先精确匹配
        if overlap > best_score:
            best_score = overlap
            best_key = key
            best_content = content

    if best_score > 0:
        return f"[来源: {best_key}]\n{best_content}"
    else:
        # 返回所有可能的条目（LLM 可以自己筛选）
        all_entries = []
        for key, content in KNOWLEDGE_BASE.items():
            all_entries.append(f"[来源: {key}]\n{content}")
        return "\n\n---\n\n".join(all_entries)


# ══════════════════════════════════════════════════════════════════════
# XML Agent 工具描述（标准 dict 格式，非 OpenAI tools 参数格式）
# ══════════════════════════════════════════════════════════════════════

XML_TOOLS = [
    {
        "name": "search_knowledge",
        "description": "在知识库中搜索信息。可以查询诺贝尔奖、AI 历史、Python 知识等。",
        "parameters": {"query": "string - 搜索查询词"},
    },
    {
        "name": "get_weather",
        "description": "查询指定城市的天气信息",
        "parameters": {"city": "string - 城市名称（中文或英文）"},
    },
]


def xml_tool_executor(name: str, args: dict) -> str:
    """XML Agent 的工具执行器。

    根据工具名称分发到对应的实现函数。
    """
    if name == "search_knowledge":
        query = args.get("query", "")
        return static_search(query)
    elif name == "get_weather":
        city = args.get("city", "上海")
        try:
            tool = WeatherTool()
            return tool.execute(city=city)
        except Exception as e:
            return f"[天气查询失败] {e}"
    else:
        return f"未知工具: {name}"


# ══════════════════════════════════════════════════════════════════════
# Demo 1: Self-Ask Agent —— 多跳推理
# ══════════════════════════════════════════════════════════════════════

def demo_self_ask():
    """演示 Self-Ask Agent 的追问→搜索→再追问循环。

    选了一个需要多步查找的问题，展示 Self-Ask 如何逐步缩小范围。
    """
    print("\n" + "=" * 60)
    print("Demo 1: Self-Ask Agent —— 纯文本模式的追问-搜索-回答")
    print("=" * 60)
    print()
    print("核心机制:")
    print("  LLM 输出 \"Follow up: <查询>\" → Agent 搜索 →")
    print("  反馈 \"Intermediate answer: <结果>\" →")
    print("  LLM 继续追问或输出 \"So the final answer is: <答案>\"")
    print()
    print("关键特征: 不依赖 OpenAI Function Calling，")
    print("          任何 LLM 都可以使用此模式。")

    llm = LLMClient(Config.llm)
    agent = SelfAskAgent(llm, search_fn=static_search, max_follow_ups=5)

    question = (
        "2024年诺贝尔物理学奖颁给了谁？"
        "获奖者的主要贡献分别是什么？"
        "这些贡献对人工智能领域产生了什么影响？"
    )

    print(f"\n[问题] {question}\n")
    result = agent.run(question)

    print(f"\n{'='*60}")
    print("最终回答:")
    print(f"{'='*60}")
    print(result)


# ══════════════════════════════════════════════════════════════════════
# Demo 2: XML Agent —— XML 标签驱动工具调用
# ══════════════════════════════════════════════════════════════════════

def demo_xml_agent():
    """演示 XML Agent 如何使用 XML 标签格式调用工具。

    与 Self-Ask 对比: XML Agent 的工具调用格式更结构化，
    可以携带多参数，更接近 Function Calling 的效果。
    """
    print("\n" + "=" * 60)
    print("Demo 2: XML Agent —— XML 标签格式调用工具")
    print("=" * 60)
    print()
    print("核心机制:")
    print("  LLM 输出 <tool_call><name>工具名</name><arguments>参数</arguments></tool_call>")
    print("  → Agent 解析 XML 执行工具 →")
    print("  反馈 <tool_result><name>工具名</name><result>结果</result></tool_result>")
    print("  → LLM 根据结果继续推理或给出最终答案")
    print()
    print("对比 Function Calling:")
    print("  FC: OpenAI API tools 参数 → tool_calls JSON")
    print("  XML: System Prompt 描述工具 → LLM 输出 XML 文本")

    llm = LLMClient(Config.llm)
    agent = XMLAgent(
        llm,
        tools=XML_TOOLS,
        tool_executor=xml_tool_executor,
        max_iterations=4,
    )

    task = (
        "查询上海现在的天气，然后在知识库中搜索关于 Python 语言历史的信息。"
        "最后根据天气和 Python 知识，给出一个用 Python 写天气分析脚本的建议。"
    )

    print(f"\n[任务] {task}\n")
    result = agent.run(task)

    print(f"\n{'='*60}")
    print("最终回答:")
    print(f"{'='*60}")
    print(result)


# ══════════════════════════════════════════════════════════════════════
# Demo 3: Conversational Agent —— 现有 Agent + Memory
# ══════════════════════════════════════════════════════════════════════

def demo_conversational():
    """演示 Conversational Agent —— 其实就是现有的 ReAct Agent 加上 Memory。

    不需要新增任何代码，Agent + Memory 已经构成了完整的 Conversational Agent。
    """
    print("\n" + "=" * 60)
    print("Demo 3: Conversational Agent = ReAct Agent + Memory")
    print("=" * 60)
    print()
    print("LangChain 中的 ConversationalAgent 本质上就是:")
    print("  ReAct Agent + ConversationBufferMemory")
    print()
    print("我们的 Agent 从一开始就设计了 Memory 接口，")
    print("因此不需要额外实现 —— 加上 Memory 就是 Conversational Agent。")

    llm = LLMClient(Config.llm)
    registry = ToolRegistry()
    registry.register(WeatherTool())

    # 带记忆的 Agent = Conversational Agent
    memory = ConversationBufferMemory()
    agent = Agent(llm_client=llm, registry=registry, memory=memory)

    print("\n[第 1 轮对话]")
    print("User: 上海今天天气怎么样？")
    agent.run("上海今天天气怎么样？")

    print("\n[第 2 轮对话]")
    print("User: 那北京呢？（依赖上一轮上下文）")
    result = agent.run("那北京呢？")

    print(f"\n[回复] {result}")


# ══════════════════════════════════════════════════════════════════════
# Demo 4: 三种 Agent 格式对比
# ══════════════════════════════════════════════════════════════════════

def demo_format_comparison():
    """对比三种 Agent 类型的工具调用格式差异。"""
    print("\n" + "=" * 60)
    print("Demo 4: 工具调用格式对比")
    print("=" * 60)
    print()

    table = """
+------------------+------------------------------------+---------------------+
| Agent 类型        | 工具调用格式                         | 适用模型             |
+------------------+------------------------------------+---------------------+
| ReAct Agent      | OpenAI tools 参数                   | 支持 FC 的模型       |
| (Function Call)  | → tool_calls JSON                   | (GPT/DeepSeek 等)   |
+------------------+------------------------------------+---------------------+
| Self-Ask Agent   | Follow up: <查询>                   | 任何模型             |
| (纯文本)          | (由 Agent 解析并搜索)               |                     |
+------------------+------------------------------------+---------------------+
| XML Agent        | <tool_call>                         | 任何模型             |
| (XML 标签)        |   <name>工具</name>                 |                     |
|                  |   <arguments>JSON</arguments>       |                     |
|                  | </tool_call>                        |                     |
+------------------+------------------------------------+---------------------+
| Conversational   | 同 ReAct Agent                      | 支持 FC 的模型       |
| (ReAct+Memory)   | + 自动加载历史消息                   |                     |
+------------------+------------------------------------+---------------------+

选择建议:
  - 模型支持 Function Calling → 优先使用 ReAct Agent（最可靠）
  - 模型不支持 FC（开源模型）→ 使用 XML Agent（结构化，支持多参数）
  - 简单搜索型任务 → Self-Ask Agent（最简洁，易于调试）
  - 多轮对话场景 → 给任意 Agent 加上 Memory 即可
"""
    print(table)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    """交互式入口，通过命令触发不同演示。"""
    print("=" * 50)
    print("第 8b 阶段: Self-Ask / XML / Conversational Agent 演示")
    print("=" * 50)
    print()
    print("命令:")
    print("  /self-ask        运行 Self-Ask Agent 演示（多跳推理）")
    print("  /xml             运行 XML Agent 演示（XML 标签调用工具）")
    print("  /conversational  运行 Conversational Agent 演示（ReAct + Memory）")
    print("  /compare         运行三种 Agent 格式对比")
    print("  /all             运行所有演示")
    print("  /quit            退出")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[再见!]")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("[再见!]")
            break
        if user_input.lower() == "/self-ask":
            demo_self_ask()
            continue
        if user_input.lower() == "/xml":
            demo_xml_agent()
            continue
        if user_input.lower() == "/conversational":
            demo_conversational()
            continue
        if user_input.lower() == "/compare":
            demo_format_comparison()
            continue
        if user_input.lower() == "/all":
            demo_self_ask()
            print("\n")
            demo_xml_agent()
            print("\n")
            demo_conversational()
            print("\n")
            demo_format_comparison()
            print("\n" + "=" * 60)
            print("所有演示完成！")
            print("=" * 60)
            continue

        print(f"未知命令: {user_input}")
        print("可用命令: /self-ask, /xml, /conversational, /compare, /all, /quit")


if __name__ == "__main__":
    main()

