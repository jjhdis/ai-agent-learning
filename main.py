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
    /demo-parser    运行输出解析系统演示
    /demo-stream    运行流式输出演示
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
from agent.output_parsers.str_parser import StrOutputParser
from agent.output_parsers.json_parser import JsonOutputParser
from agent.output_parsers.pydantic_parser import PydanticOutputParser
from agent.output_parsers.list_parser import CommaSeparatedListOutputParser
from agent.output_parsers.base import OutputParserException
from agent.streaming.event import StreamEventType, StreamEvent, StreamAccumulator
from agent.streaming.handler import StreamingCallbackHandler
from agent.rag.retriever import RetrieverTool


# ──────────────────────────────────────────────
# 组装
# ──────────────────────────────────────────────

def build_agent(memory=None, callbacks=None, output_parser=None):
    """组装 Agent：配置 LLM + 注册工具 + 可选记忆 + 可选回调 + 可选输出解析器。"""
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
        output_parser=output_parser,
    )


# ──────────────────────────────────────────────
# RAG 演示
# ──────────────────────────────────────────────

def demo_rag(agent_base):
    """演示 RAG (检索增强生成) 完整管道:
    Document → Splitter → Embedding → VectorStore → RetrieverTool → Agent 调用。
    """
    print("\n" + "=" * 60)
    print("RAG 演示 —— 检索增强生成 (Retrieval-Augmented Generation)")
    print("=" * 60)

    # ── 1. 准备知识库文档 ──
    print("\n" + "─" * 50)
    print("1. 准备知识库文档 —— 构建 Python 知识点文档集")
    print("─" * 50)

    knowledge_texts = [
        # Python 基础
        (
            "Python 由 Guido van Rossum 于 1991 年首次发布。"
            "它是一种解释型、高级、通用型编程语言，设计哲学强调代码可读性和简洁的语法。"
            "Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。"
            "Python 使用缩进而不是大括号来划分代码块，这使得代码更加整洁一致。"
            "Python 是动态类型语言，变量在赋值时自动确定类型，无需显式声明。"
        ),
        # GIL
        (
            "GIL (Global Interpreter Lock，全局解释器锁) 是 CPython 中的一种互斥锁。"
            "它确保同一时刻只有一个线程在执行 Python 字节码。"
            "GIL 的存在简化了 CPython 的内存管理（特别是引用计数），但也限制了多线程在 CPU 密集型任务上的性能。"
            "对于 I/O 密集型任务（如网络请求、文件读写），GIL 的影响较小，因为线程在等待 I/O 时会释放 GIL。"
            "要绕过 GIL 的限制，可以使用多进程（multiprocessing 模块）或异步编程（asyncio）。"
            "其他 Python 实现（如 Jython、IronPython）没有 GIL。"
            "Python 3.13 引入了 PEP 703，计划在未来版本中移除 GIL。"
        ),
        # 装饰器
        (
            "装饰器 (Decorator) 是 Python 中一种强大的设计模式，本质上是接受函数作为参数并返回新函数的高阶函数。"
            "使用 @decorator_name 语法糖可以简洁地将装饰器应用于函数或类。"
            "常见的装饰器有 @staticmethod、@classmethod、@property、@functools.lru_cache 等。"
            "装饰器可以用于日志记录、性能计时、权限校验、缓存、路由注册等场景。"
            "多个装饰器可以叠加使用，执行顺序是从下往上（靠近函数定义的先执行）。"
            "functools.wraps 装饰器用于保留原函数的元数据（如 __name__、__doc__）。"
        ),
        # 生成器
        (
            "生成器 (Generator) 是一种使用 yield 关键字的特殊迭代器。"
            "与普通函数不同，生成器在执行到 yield 语句时会暂停并保存当前状态，下次调用时从暂停处继续执行。"
            "生成器是惰性求值的，只在需要时才产生值，这使得它们非常内存友好，特别适合处理大型数据集。"
            "生成器表达式 (generator expression) 使用类似列表推导式的语法但用圆括号，如 (x**2 for x in range(10))。"
            "yield from 语句可以将一个生成器的所有值委托给另一个生成器。"
            "生成器常用于数据流处理、无限序列生成、协程等场景。"
        ),
        # asyncio
        (
            "asyncio 是 Python 3.4+ 引入的标准库，用于编写单线程并发代码。"
            "它基于事件循环 (event loop)，使用 async/await 语法定义协程 (coroutine)。"
            "await 关键字用于暂停当前协程的执行，等待另一个协程或 Future 对象完成。"
            "asyncio.gather() 可以并发运行多个协程并收集结果。"
            "asyncio 特别适合网络编程、Web 服务等高 I/O 并发场景。"
            "与多线程相比，asyncio 避免了线程切换开销和 GIL 竞争，但需要所有 I/O 操作都是非阻塞的。"
            "Python 3.11+ 引入了 asyncio.TaskGroup() 提供更好的任务管理。"
        ),
        # 类型提示
        (
            "Python 3.5+ 引入了类型提示 (Type Hints)，允许在代码中标注变量、函数参数和返回值的类型。"
            "类型提示是可选的，不会影响运行时行为，但可以被 IDE、类型检查器（如 mypy、pyright）和文档工具使用。"
            "typing 模块提供了丰富类型: List、Dict、Optional、Union、Callable、Protocol、TypedDict 等。"
            "Python 3.9+ 可以直接使用 list[int]、dict[str, int] 替代 typing.List[int] 等。"
            "Python 3.10+ 引入了 | 联合类型语法: int | None 替代 Optional[int]。"
            "Python 3.12+ 引入了类型参数语法的简化: def foo[T](x: T) -> T。"
        ),
        # pip
        (
            "pip 是 Python 的官方包管理工具，用于从 PyPI (Python Package Index) 安装和管理第三方库。"
            "常用命令: pip install package、pip uninstall package、pip list、pip freeze > requirements.txt、pip install -r requirements.txt。"
            "pip 支持版本约束: pip install package==1.0.0、pip install 'package>=2.0,<3.0'。"
            "虚拟环境 (venv/virtualenv) 为每个项目创建独立的包环境，避免依赖冲突。"
            "Python 3.4+ 内置 venv 模块: python -m venv .venv。"
        ),
        # 列表推导式
        (
            "列表推导式 (List Comprehension) 是 Python 中创建列表的简洁语法。"
            "基本格式: [expression for item in iterable if condition]。"
            "列表推导式通常比等效的 for 循环更快，因为它们在 C 层面执行。"
            "字典推导式和集合推导式也使用类似的语法: {k: v for k, v in ...}、{x for x in ...}。"
            "嵌套推导式虽然强大但可读性可能下降，复杂逻辑建议改用传统循环。"
        ),
    ]

    metadatas = [
        {"topic": "Python 基础", "category": "语言概述"},
        {"topic": "GIL", "category": "并发"},
        {"topic": "装饰器", "category": "高级特性"},
        {"topic": "生成器", "category": "迭代器与生成器"},
        {"topic": "asyncio", "category": "并发"},
        {"topic": "类型提示", "category": "语言特性"},
        {"topic": "pip", "category": "工具与生态"},
        {"topic": "列表推导式", "category": "基础语法"},
    ]

    print(f"   知识库文档数: {len(knowledge_texts)} 篇")
    for text, meta in zip(knowledge_texts, metadatas):
        topic = meta["topic"]
        preview = text[:40]
        print(f"     📄 {topic}: {preview}…")

    # ── 2. 分割文档 ──
    print("\n" + "─" * 50)
    print("2. 文档分割 —— RecursiveCharacterTextSplitter")
    print("─" * 50)

    from agent.rag.splitter import RecursiveCharacterTextSplitter
    from agent.rag.document import Document

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, overlap_size=50)
    docs = []
    for i, text in enumerate(knowledge_texts):
        docs.append(Document(content=text, metadata=metadatas[i]))
    chunks = splitter.split_documents(docs)
    print(f"   分割前: {len(docs)} 篇文档")
    print(f"   分割后: {len(chunks)} 个片段")
    print(f"   chunk_size={splitter.chunk_size}, overlap_size={splitter.overlap_size}")
    for i, chunk in enumerate(chunks[:5]):
        print(f"   块 {i}: len={len(chunk.content)}, topic={chunk.metadata.get('topic', '?')}, preview={chunk.content[:50]}…")
    if len(chunks) > 5:
        print(f"   … 其余 {len(chunks) - 5} 个片段省略")

    # ── 3. Embedding 与向量存储 ──
    print("\n" + "─" * 50)
    print("3. 向量化与存储 —— InMemoryVectorStore + OpenAIEmbeddings")
    print("─" * 50)

    from agent.rag.store import InMemoryVectorStore
    from agent.rag.embedding import OpenAIEmbeddings

    try:
        embedding = OpenAIEmbeddings()
        print(f"   模型: {embedding.model}")
        # 用一句话测试 Embedding API 连通性
        test_vec = embedding.embed_query("测试")
        print(f"   向量维度: {len(test_vec)}")
        print(f"   前 5 维: {[round(v, 4) for v in test_vec[:5]]}")
        embedding_ok = True
    except Exception as e:
        print(f"   [警告] Embedding API 不可用: {e}")
        print(f"   [降级] 切回本地关键词匹配模式 (SimpleEmbeddings)")
        from agent.rag.embedding import SimpleEmbeddings
        embedding = SimpleEmbeddings()
        embedding_ok = False

    store = InMemoryVectorStore(embedding=embedding)
    store.add_documents(chunks)
    print(f"   存储文档块数: {store.document_count}")

    # ── 4. 创建 RetrieverTool 并注册 ──
    print("\n" + "─" * 50)
    print("4. RetrieverTool —— 封装为 Agent 可调用的工具")
    print("─" * 50)

    retriever = RetrieverTool()
    retriever.add_documents(docs)  # 重做一遍（RetrieverTool 内部有 splitter）
    print(f"   工具名: {retriever.name}")
    print(f"   参数: {[p.name for p in retriever.parameters]}")
    print(f"   知识库文档块数: {retriever.document_count}")
    print(f"   OpenAI 函数定义：")
    fn_def = retriever.to_openai_function()
    print(f"     name: {fn_def['function']['name']}")
    print(f"     description: {fn_def['function']['description'][:60]}…")
    print(f"     parameters: {list(fn_def['function']['parameters']['properties'].keys())}")

    # ── 5. 直接检索测试 ──
    print("\n" + "─" * 50)
    print("5. 直接检索测试 —— 不通过 Agent")
    print("─" * 50)

    test_queries = [
        "Python 是什么时候发布的？",
        "GIL 是什么意思？",
        "装饰器有什么用途？",
        "asyncio 怎么使用？",
        "列表推导式语法是什么？",
    ]

    for query in test_queries:
        result = retriever.execute(query)
        # 只展示第一条结果的摘要
        first_line = result.split("\n")[0] if result else "无结果"
        print(f"\n   查询: {query}")
        print(f"   {first_line}")

    # ── 6. Agent 集成演示 ──
    print("\n" + "─" * 50)
    print("6. Agent 集成 —— Agent 通过 function calling 使用知识库")
    print("─" * 50)

    registry = agent_base.registry.__class__()
    registry.register(retriever)

    from agent.core.agent import Agent
    from agent.llm.client import LLMClient
    from config import Config

    rag_agent = Agent(
        llm_client=LLMClient(Config.llm),
        registry=registry,
        hooks={
            "on_start": lambda msg: print(f"\n   [Agent 开始]: {msg}"),
            "on_tool_call": lambda name, args: print(f"   [Tool 调用]: {name}({args})"),
            "on_reply": lambda msg: print(f"   [Agent 回复]: {msg}"),
        },
    )

    questions = [
        "什么是 GIL？它对 Python 多线程有什么影响？",
        "Python 的装饰器是什么？请根据知识库内容回答。",
    ]

    for question in questions:
        print(f"\n   用户提问: {question}")
        try:
            answer = rag_agent.run(question)
            if isinstance(answer, str):
                print(f"\n   [最终回复]: {answer[:200]}…" if len(answer) > 200 else f"\n   [最终回复]: {answer}")
            else:
                print(f"\n   [最终回复]: {answer}")
        except Exception as e:
            print(f"   [Agent 错误]: {e}")

    print("\n" + "=" * 60)
    print("RAG 演示完成！")
    print("=" * 60)


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


# ──────────────────────────────────────────────
# 输出解析演示
# ──────────────────────────────────────────────

def demo_output_parsers(agent_base):
    """演示输出解析系统：StrOutputParser / JsonOutputParser / PydanticOutputParser / CommaSeparatedListOutputParser。"""
    print("\n" + "=" * 60)
    print("输出解析系统演示 —— Output Parser")
    print("=" * 60)

    # ── 1. StrOutputParser —— 原样返回 ──
    print("\n" + "─" * 50)
    print("1. StrOutputParser —— 原样返回字符串")
    print("─" * 50)

    str_parser = StrOutputParser()
    result = str_parser.parse("这是一段 LLM 回复文本")
    print(f"   输入: '这是一段 LLM 回复文本'")
    print(f"   输出: '{result}'")
    print(f"   类型: {type(result).__name__}")

    # ── 2. JsonOutputParser —— 解析 JSON ──
    print("\n" + "─" * 50)
    print("2. JsonOutputParser —— 智能提取并解析 JSON")
    print("─" * 50)

    json_parser = JsonOutputParser(expected_keys=["city", "temperature", "condition"])

    # 演示格式说明
    print("   [格式说明 (可追加到 System Prompt)]:")
    instructions = json_parser.get_format_instructions()
    for line in instructions.split("\n"):
        print(f"   | {line}")

    # 场景1: 纯 JSON
    result = json_parser.parse('{"city": "上海", "temperature": 28.0, "condition": "晴"}')
    print(f"\n   场景1 - 纯 JSON:")
    print(f"     输入: '{{\"city\": \"上海\", \"temperature\": 28.0, \"condition\": \"晴\"}}'")
    print(f"     输出: {result}")

    # 场景2: JSON 嵌在文本中
    result = json_parser.parse("好的，根据查询结果，上海的天气信息如下：{\"city\": \"北京\", \"temperature\": 22.5, \"condition\": \"多云\"}")
    print(f"\n   场景2 - JSON 嵌在文本中:")
    print(f"     输入: '好的，根据查询结果...{{\"city\": \"北京\"...}}'")
    print(f"     输出: {result}")

    # 场景3: Markdown 代码块
    result = json_parser.parse('```json\n{"city": "广州", "temperature": 30.0, "condition": "雷阵雨"}\n```')
    print(f"\n   场景3 - Markdown 代码块:")
    print(f"     输入: '```json\\n...\\n```'")
    print(f"     输出: {result}")

    # ── 3. CommaSeparatedListOutputParser ──
    print("\n" + "─" * 50)
    print("3. CommaSeparatedListOutputParser —— 解析列表文本")
    print("─" * 50)

    list_parser = CommaSeparatedListOutputParser()

    # 场景1: 逗号分隔
    result = list_parser.parse("天气查询, 翻译, 计算器, 日程管理")
    print(f"   场景1 - 逗号分隔: {result}")

    # 场景2: 中文逗号
    result = list_parser.parse("上海，北京，广州，深圳")
    print(f"   场景2 - 中文逗号: {result}")

    # 场景3: 编号列表
    result = list_parser.parse("1. 查询天气\n2. 预订酒店\n3. 规划路线\n4. 推荐美食")
    print(f"   场景3 - 编号列表: {result}")

    # 场景4: JSON 数组
    result = list_parser.parse('["数据分析", "机器学习", "深度学习", "NLP"]')
    print(f"   场景4 - JSON 数组: {result}")

    print(f"\n   [格式说明]:")
    instructions = list_parser.get_format_instructions()
    print(f"   {instructions}")

    # ── 4. PydanticOutputParser ──
    print("\n" + "─" * 50)
    print("4. PydanticOutputParser —— 解析为 Pydantic 模型")
    print("─" * 50)

    try:
        from pydantic import BaseModel, Field

        class WeatherInfo(BaseModel):
            """天气信息模型"""
            city: str = Field(description="城市名称")
            temperature: float = Field(description="温度（摄氏度）")
            condition: str = Field(description="天气状况，如晴、多云、雨")

        pydantic_parser = PydanticOutputParser(pydantic_object=WeatherInfo)

        print("   [自动生成的格式说明 (从 Pydantic 模型推导)]:")
        instructions = pydantic_parser.get_format_instructions()
        for line in instructions.split("\n"):
            print(f"   | {line}")

        result = pydantic_parser.parse('{"city": "深圳", "temperature": 29.5, "condition": "晴间多云"}')
        print(f"\n   解析结果: {result}")
        print(f"   类型: {type(result).__name__}")
        print(f"   访问字段: .city={result.city}, .temperature={result.temperature}°C, .condition={result.condition}")

    except ImportError as e:
        print(f"   [跳过] Pydantic 未安装，跳过 PydanticOutputParser 演示。")
        print(f"   安装方法: pip install pydantic")

    # ── 5. 输出解析器作为 Runnable 管道组件 ──
    print("\n" + "─" * 50)
    print("5. Runnable 协议 —— 解析器作为管道组件")
    print("─" * 50)

    from agent.chain.passthrough import RunnableLambda

    # 模拟一个返回 JSON 字符串的步骤
    mock_llm = RunnableLambda(lambda x: '{"city": "成都", "temperature": 24.0, "condition": "阴"}')

    pipeline = mock_llm | JsonOutputParser(expected_keys=["city", "temperature", "condition"])
    result = pipeline.invoke("成都天气怎么样？")
    print(f"   管道: MockLLM | JsonOutputParser")
    print(f"   输入: '成都天气怎么样？'")
    print(f"   输出: {result}")
    print(f"   类型: {type(result).__name__}")

    # ── 6. Agent 集成演示 ──
    print("\n" + "─" * 50)
    print("6. Agent 集成 —— 带输出解析器的 Agent")
    print("─" * 50)

    json_agent = build_agent(output_parser=JsonOutputParser(
        expected_keys=["city", "temperature", "condition", "summary"]
    ))
    print("   [创建了带 JsonOutputParser 的 Agent]")
    print("   Agent 的最终回复将自动解析为 Python dict...")
    result = json_agent.run(
        "请以JSON格式告诉我上海今天的天气情况，"
        "包含以下字段：city（城市）、temperature（温度）、"
        "condition（天气状况）、summary（简短总结）。不要添加其他文字。"
    )
    print(f"\n   解析后结果类型: {type(result).__name__}")
    print(f"   解析后数据:")
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"      {key}: {value}")
    else:
        print(f"      {result}")

    # ── 7. 异常处理演示 ──
    print("\n" + "─" * 50)
    print("7. 异常处理 —— OutputParserException")
    print("─" * 50)

    parser = JsonOutputParser(expected_keys=["name", "age"])

    # 正常解析
    try:
        result = parser.parse('{"name": "张三", "age": 25}')
        print(f"   正常解析: {result}")
    except OutputParserException as e:
        print(f"   异常: {e}")

    # 无法解析的文本
    try:
        result = parser.parse("这是一段完全不是 JSON 的文本，没有任何花括号")
    except OutputParserException as e:
        print(f"   解析失败: {e}")
        print(f"   异常类型: {type(e).__name__}")
        print(f"   携带原始文本前80字: {e.text[:80]}")

    # ── 8. 综合对比 ──
    print("\n" + "─" * 50)
    print("8. 解析器对比总结")
    print("─" * 50)

    print(f"   {'解析器':<35} {'输入示例':<30} {'输出类型':<15}")
    print(f"   {'─'*35} {'─'*30} {'─'*15}")
    json_example = '{"key": "value"}'
    pydantic_example = '{"city": "..."}'
    print(f"   {'StrOutputParser':<35} {'任意文本':<30} {'str':<15}")
    print(f"   {'JsonOutputParser':<35} {json_example:<30} {'dict':<15}")
    print(f"   {'PydanticOutputParser':<35} {pydantic_example:<30} {'WeatherInfo':<15}")
    print(f"   {'CommaSeparatedListOutputParser':<35} {'A, B, C':<30} {'list[str]':<15}")

    print("\n" + "=" * 60)
    print("输出解析系统演示完成！")
    print("=" * 60)


# ──────────────────────────────────────────────
# 流式输出演示
# ──────────────────────────────────────────────

def demo_streaming(agent_base):
    """演示流式输出：LLM 流式 / StreamEvent 类型 / StreamAccumulator / Agent.run_stream()。"""
    print("\n" + "=" * 60)
    print("流式输出演示 —— Streaming")
    print("=" * 60)

    # ── 1. LLM 流式调用基础 ──
    print("\n" + "─" * 50)
    print("1. LLMClient.stream_chat() —— 逐 Token 流式输出")
    print("─" * 50)

    messages = [
        {"role": "user", "content": "用一句话介绍 Python 语言的特点（30字以内）。"}
    ]
    print("   [实时流式输出]:")
    print("   ", end="", flush=True)
    for chunk in agent_base.llm.stream_chat(messages):
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print()

    # ── 2. StreamAccumulator 使用 ──
    print("\n" + "─" * 50)
    print("2. StreamAccumulator —— 流式块重组为完整消息")
    print("─" * 50)

    acc = StreamAccumulator()
    token_count = 0
    for chunk in agent_base.llm.stream_chat(messages):
        acc.add_chunk(chunk)
        new_tokens = acc.new_tokens()
        if new_tokens:
            token_count += len(new_tokens)

    print(f"   累积的完整内容 ({token_count} 字符):")
    print(f"   \"{acc.content}\"")
    print(f"   是否含工具调用: {acc.has_tool_calls}")
    print(f"   是否已完成: {acc.is_done}")
    print(f"   finish_reason: {acc.finish_reason}")

    # ── 3. StreamEvent 类型 ──
    print("\n" + "─" * 50)
    print("3. StreamEvent 类型一览")
    print("─" * 50)

    events_demo = [
        StreamEvent(StreamEventType.THINK, "今", step=0),
        StreamEvent(StreamEventType.THINK, "天", step=0),
        StreamEvent(StreamEventType.TOOL_START, {"name": "get_weather", "args": {"city": "上海"}}, step=0),
        StreamEvent(StreamEventType.TOOL_END, {"name": "get_weather", "result": "上海晴，28°C"}, step=0),
        StreamEvent(StreamEventType.REPLY, "上", step=1),
        StreamEvent(StreamEventType.REPLY, "海", step=1),
        StreamEvent(StreamEventType.DONE, "上海晴，28°C", step=1),
    ]
    for evt in events_demo:
        print(f"   {evt}")

    # ── 4. StreamingCallbackHandler ──
    print("\n" + "─" * 50)
    print("4. StreamingCallbackHandler —— 流式回调处理")
    print("─" * 50)

    handler = StreamingCallbackHandler(mode="verbose", show_thinking=False)
    print(f"   Handler: {handler!r}")
    for evt in events_demo:
        handler.handle_event(evt)
    print(f"\n   统计: tokens={handler.token_count}, tools={handler.tool_count}")

    # ── 5. Agent.run_stream() 完整演示 ──
    print("\n" + "─" * 50)
    print("5. Agent.run_stream() —— 带工具的流式 Agent")
    print("─" * 50)

    stream_agent = build_agent()
    print("   [流式 Agent 启动，查询上海天气...]")
    print("   ", end="", flush=True)

    handler = StreamingCallbackHandler(mode="realtime", show_thinking=False)
    for event in stream_agent.run_stream("上海今天天气怎么样？"):
        handler.handle_event(event)

    print(f"\n   统计: tokens={handler.token_count}, tools={handler.tool_count}, "
          f"llm_calls={handler.llm_call_count}, elapsed={handler.elapsed:.2f}s")

    # ── 6. 对比 run() vs run_stream() ──
    print("\n" + "─" * 50)
    print("6. run() vs run_stream() 对比")
    print("─" * 50)

    print(f"   {'特性':<25} {'run()':<30} {'run_stream()':<30}")
    print(f"   {'─'*25} {'─'*30} {'─'*30}")
    print(f"   {'返回方式':<25} {'一次性返回完整结果':<30} {'逐 Token 流式产出':<30}")
    print(f"   {'返回类型':<25} {'解析后的数据 (Any)':<30} {'生成器 yield StreamEvent':<30}")
    print(f"   {'用户体验':<25} {'等待全部生成完':<30} {'实时逐字显示':<30}")
    print(f"   {'工具调用':<25} {'内部静默执行':<30} {'产出 TOOL_START/END 事件':<30}")
    print(f"   {'适用场景':<25} {'批处理/API 调用':<30} {'终端交互/聊天界面':<30}")

    # ── 7. 流式 + 输出解析 组合演示 ──
    print("\n" + "─" * 50)
    print("7. 流式 + 输出解析 组合 —— run_stream() + JsonOutputParser")
    print("─" * 50)

    json_stream_agent = build_agent(
        output_parser=JsonOutputParser(expected_keys=["city", "temperature", "condition"])
    )
    print("   [流式 Agent + JsonOutputParser，查询上海天气...]")
    print("   ", end="", flush=True)

    done_data = None
    for event in json_stream_agent.run_stream(
        "请以JSON格式告诉我上海今天的天气情况，"
        "包含以下字段：city、temperature、condition。只输出JSON不要其他文字。"
    ):
        if event.event == StreamEventType.REPLY:
            print(event.data, end="", flush=True)
        elif event.event == StreamEventType.DONE:
            done_data = event.data

    if done_data is not None:
        print(f"\n   DONE 事件携带的解析后数据: {done_data}")
        print(f"   数据类型: {type(done_data).__name__}")
        if isinstance(done_data, dict):
            for k, v in done_data.items():
                print(f"      {k}: {v}")

    # ── 8. 流式思考过程展示 ──
    print("\n" + "─" * 50)
    print("8. show_thinking=True —— 显示推理过程")
    print("─" * 50)

    think_agent = build_agent()
    think_handler = StreamingCallbackHandler(mode="realtime", show_thinking=True)
    print("   [流式 Agent (show_thinking=True)，思考内容会以灰色 token 显示]")
    for event in think_agent.run_stream("1+1等于几？直接回答。"):
        think_handler.handle_event(event)
    print()

    print("\n" + "=" * 60)
    print("流式输出演示完成！")
    print("=" * 60)


def main():
    memory = ConversationBufferMemory()
    memory_label = "完整记忆 (Buffer)"

    print("=" * 50)
    print(f"[AI Agent 已启动]")
    print(f"   模型: {Config.llm.model}")
    print(f"   工具: 天气查询 (get_weather)")
    print(f"   记忆: {memory_label}")
    print(f"   输入 /quit 退出, /reset 清空, /demo-memory 演示记忆, /demo-callback 演示回调, /demo-token 演示Token计数, /demo-prompt 演示模板, /demo-parser 演示输出解析, /demo-stream 演示流式输出, /demo-rag 演示RAG知识检索, /memory 切换记忆模式")
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
        if user_input.lower() == "/demo-parser":
            demo_output_parsers(agent)
            continue
        if user_input.lower() == "/demo-stream":
            demo_streaming(agent)
            continue
        if user_input.lower() == "/demo-rag":
            demo_rag(agent)
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
