# 第五阶段总结：输出解析（Output Parsing）

> **学习时间**：2026-05-18
> **核心问题**：LLM 返回的是非结构化文本，如何自动转换为可编程的结构化数据？

---

## 目录

1. [为什么需要输出解析？](#1-为什么需要输出解析)
2. [四个核心解析器详解](#2-四个核心解析器详解)
3. [解析器关系图](#3-解析器关系图)
4. [LangChain 的输出解析体系](#4-langchain-的输出解析体系)
5. [与 Agent 的集成：自动解析最终回复](#5-与-agent-的集成自动解析最终回复)
6. [当前代码 vs LangChain 对比](#6-当前代码-vs-langchain-对比)

---

## 1. 为什么需要输出解析？

### 1.1 当前代码的局限

```python
# 当前 Agent.run()：返回纯文本
return msg.content  # "上海今天28°C，晴"
```

问题：
- **下游无法直接使用**：调用方拿到字符串，还得手动解析
- **格式不稳定**：LLM 每次返回的格式可能不同（有的加说明，有的嵌在 Markdown 里）
- **无法验证**：返回内容是否包含必要字段？类型是否正确？
- **无法管道化**：纯文本无法直接传给下一个处理步骤

### 1.2 输出解析解决什么问题

| 问题 | 解析方案 |
|------|----------|
| 返回纯文本 | `StrOutputParser` 原样返回（保持兼容） |
| 需要字典/JSON | `JsonOutputParser` 自动提取并解析 JSON |
| 需要强类型对象 | `PydanticOutputParser` 解析为 Pydantic 模型 |
| 需要列表 | `CommaSeparatedListOutputParser` 解析各种列表格式 |
| 格式不统一 | 每个解析器都提供 `get_format_instructions()` 引导 LLM |
| 解析失败 | `OutputParserException` 携带原始文本，方便调试 |

---

## 2. 四个核心解析器详解

### 2.1 BaseOutputParser —— 抽象基类

**文件**：`agent/output_parsers/base.py`

**本质**：定义解析器的统一契约。所有解析器继承此类并实现 `parse()` 方法。

```python
class BaseOutputParser(Runnable):
    def parse(self, text: str) -> Any:          # 必须实现
        raise NotImplementedError

    def get_format_instructions(self) -> str:    # 可选，返回格式引导说明
        return ""

    def invoke(self, input_data) -> Any:         # Runnable 协议
        # 自动处理 str / dict / 对象 多种输入格式
```

**核心设计决策**：

- **继承 Runnable**：解析器天然可以作为管道组件（`prompt | llm | parser`）
- **`get_format_instructions()` 是精髓**：解析器不只是"事后解析"，还能"事前引导"——把格式说明注入 System Prompt，让 LLM 按照预期格式输出
- **`invoke()` 智能适配输入**：无论是字符串、dict、还是带 `.content` 属性的消息对象，都能正确处理

---

### 2.2 StrOutputParser —— 字符串直通解析器

**文件**：`agent/output_parsers/str_parser.py`

**本质**：最简单的解析器，原样返回 LLM 的文本输出，不做任何转换。

```python
parser = StrOutputParser()
result = parser.parse("上海今天28°C，晴")
# → "上海今天28°C，晴"  (str)
```

**定位**：作为 Agent 的默认解析器，保证 `Agent.run()` 的返回类型向后兼容。

---

### 2.3 JsonOutputParser —— JSON 提取解析器

**文件**：`agent/output_parsers/json_parser.py`

**本质**：从 LLM 的文本输出中智能提取 JSON 对象并解析为 Python dict。

**处理策略（按优先级）**：

```
输入文本
  ├─ 1. 直接 json.loads(text)            → 纯 JSON
  ├─ 2. 提取 Markdown 代码块内容          → ```json ... ```
  ├─ 3. 括号计数提取第一个完整 JSON 对象   → 文本中嵌的 {...}
  └─ 4. 修复常见错误后重试                 → None→null, True→true 等
```

**健壮性设计**：

```python
# 场景1: 纯 JSON
parser.parse('{"city": "上海", "temperature": 28.0}')

# 场景2: JSON 嵌在文本中
parser.parse('根据查询结果，天气信息为：{"city": "上海", "temperature": 28.0}')

# 场景3: Markdown 代码块
parser.parse('```json\n{"city": "上海", "temperature": 28.0}\n```')
```

三种场景均返回 `{"city": "上海", "temperature": 28.0}`。

**`get_format_instructions()` 生成引导词**：

```
请严格按以下 JSON 格式输出，不要添加 Markdown 代码块标记或其他文字:
{
    "city": <city的值>
    "temperature": <temperature的值>
    "condition": <condition的值>
}
```

---

### 2.4 PydanticOutputParser —— 强类型模型解析器

**文件**：`agent/output_parsers/pydantic_parser.py`

**本质**：将 JSON 解析为指定的 Pydantic BaseModel 实例，提供编译时级别的类型安全。

```python
from pydantic import BaseModel, Field

class WeatherInfo(BaseModel):
    city: str = Field(description="城市名称")
    temperature: float = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况")

parser = PydanticOutputParser(pydantic_object=WeatherInfo)
result = parser.parse('{"city": "上海", "temperature": 28.0, "condition": "晴"}')

print(result.city)           # "上海"  ← 属性访问，IDE 补全
print(type(result))          # <class 'WeatherInfo'>
```

**核心优势**：

| 特性 | JsonOutputParser | PydanticOutputParser |
|------|-----------------|---------------------|
| 输出类型 | `dict` | `WeatherInfo` 实例 |
| 字段验证 | 无（只检查是合法 JSON） | 自动验证类型、必填 |
| IDE 支持 | `result["city"]` 字符串键 | `result.city` 点号 + 自动补全 |
| 格式说明 | 手动指定 `expected_keys` | 从模型自动生成 JSON Schema |

**自动生成格式说明**（从模型字段推导）：

```
请严格按以下 JSON 格式输出（WeatherInfo），不要添加其他文字:
{
  "city": <string> 【必填】  // 城市名称
  "temperature": <number> 【必填】  // 温度（摄氏度）
  "condition": <string> 【必填】  // 天气状况
}
```

**兼容性**：同时支持 Pydantic v1（`.schema()`）和 v2（`.model_json_schema()`）。

---

### 2.5 CommaSeparatedListOutputParser —— 列表解析器

**文件**：`agent/output_parsers/list_parser.py`

**本质**：将 LLM 输出的列表文本解析为 `list[str]`，支持多种常见格式。

**支持的格式**：

```python
parser = CommaSeparatedListOutputParser()

# 英文逗号分隔
parser.parse("weather, translate, calculator")  # → ["weather", "translate", "calculator"]

# 中文逗号分隔
parser.parse("天气，翻译，计算器")                 # → ["天气", "翻译", "计算器"]

# 编号列表
parser.parse("1. 天气查询\n2. 翻译\n3. 计算")    # → ["天气查询", "翻译", "计算"]

# JSON 数组
parser.parse('["数据分析", "机器学习", "深度学习"]') # → ["数据分析", "机器学习", "深度学习"]
```

**使用场景**：让 LLM 推荐一组选项、列出步骤、枚举关键词等。

---

## 3. 解析器关系图

```
BaseOutputParser (ABC + Runnable)
├── StrOutputParser          ← 默认，原样返回字符串
├── JsonOutputParser         ← 智能提取 + 解析 JSON → dict
│   └── 括号计数 / 正则 / 错误修复
├── PydanticOutputParser     ← JSON → Pydantic 模型实例
│   └── 自动生成 JSON Schema 格式说明
└── CommaSeparatedListOutputParser ← 文本 → list[str]
    └── 多格式识别（逗号/编号/JSON数组）

OutputParserException        ← 解析失败时抛出，携带原始文本
```

**设计原则**：

1. **基类提供 Runnable 协议**：所有解析器天然可组合到管道中
2. **`get_format_instructions()` 独立于 `parse()`**：引导是"事前"的，解析是"事后"的，职责分离
3. **解析器不依赖 Agent**：可以独立使用，也可以在管道中组合
4. **健壮优先**：JsonOutputParser 不假设 LLM 输出一定是纯净 JSON，而是尽可能从中提取

---

## 4. LangChain 的输出解析体系

### 4.1 LangChain 提供的解析器

| LangChain 解析器 | 我们的实现 | 说明 |
|---|---|---|
| `StrOutputParser` | `StrOutputParser` ✅ | 完全一致 |
| `JsonOutputParser` | `JsonOutputParser` ✅ | 核心逻辑一致 |
| `PydanticOutputParser` | `PydanticOutputParser` ✅ | 兼容 v1/v2 |
| `CommaSeparatedListOutputParser` | `CommaSeparatedListOutputParser` ✅ | 多格式支持 |
| `StructuredOutputParser` | 未实现 | 不需 Pydantic 的结构化方案 |
| `EnumOutputParser` | 未实现 | 枚举值解析 |
| `OutputFixingParser` | 未实现 | 解析失败时用 LLM 修复 |
| `RetryOutputParser` | 未实现 | 重试机制 |

### 4.2 学习收获

通过手写这四个解析器，理解了 LangChain 的设计思路：

- **`get_format_instructions()` 是精髓**：输出解析不只是"事后处理"，而是和 Prompt 配合的"事前约束"。把格式说明注入 Prompt，让 LLM 按照约定格式输出，解析成功率大幅提升
- **Runnable 协议让一切可组合**：解析器实现 `invoke()` 后，`prompt | llm | parser` 就是一条完整的处理链
- **健壮性 > 严格性**：JsonOutputParser 的括号计数、Markdown 提取、错误修复，都是为了让"不完美的 LLM 输出"也能被正确解析

---

## 5. 与 Agent 的集成：自动解析最终回复

### 5.1 集成方式

在 `Agent.__init__` 中增加 `output_parser` 参数，默认为 `StrOutputParser()`：

```python
class Agent:
    def __init__(self, ..., output_parser: BaseOutputParser = None):
        self.output_parser = output_parser or StrOutputParser()

    def run(self, user_message: str):
        # ... ReAct 循环 ...
        final_reply = msg.content
        return self.output_parser.parse(final_reply)  # 解析后返回
```

### 5.2 使用示例

```python
# 默认行为：返回字符串（向后兼容）
agent = Agent(llm_client=llm, registry=registry)
result = agent.run("上海天气怎么样？")
# → "上海今天28°C，晴"  (str)

# 使用 JsonOutputParser：返回 dict
agent = Agent(
    llm_client=llm,
    registry=registry,
    output_parser=JsonOutputParser(["city", "temperature", "condition"]),
)
result = agent.run("以JSON格式告诉我上海天气")
# → {"city": "上海", "temperature": 28.0, "condition": "晴"}  (dict)
```

### 5.3 配合 System Prompt 使用（最佳实践）

```python
parser = JsonOutputParser(expected_keys=["city", "temperature", "condition"])

# 将格式说明注入 System Prompt
system_prompt = (
    "你是一个天气助手。回答用户关于天气的问题。\n"
    + parser.get_format_instructions()
)

# LLM 输出会自动遵循 JSON 格式，解析器再将其转为 dict
result = agent.run("上海天气")
# → {"city": "上海", "temperature": 28.0, "condition": "晴"}
```

---

## 6. 当前代码 vs LangChain 对比

### 6.1 功能对比

| 功能 | 当前代码 | LangChain |
|------|---------|-----------|
| 基础解析器 | `StrOutputParser` | `StrOutputParser` |
| JSON 解析 | `JsonOutputParser`（括号计数 + 正则 + 修复） | `JsonOutputParser`（类似逻辑） |
| Pydantic 解析 | `PydanticOutputParser`（v1/v2 兼容） | `PydanticOutputParser` |
| 列表解析 | `CommaSeparatedListOutputParser`（多格式） | `CommaSeparatedListOutputParser` |
| 格式说明生成 | `get_format_instructions()` | `get_format_instructions()` |
| 管道组合 | Runnable 协议（`invoke()`） | LCEL（`\|` 操作符） |
| 异常定义 | `OutputParserException` | `OutputParserException` |
| Agent 集成 | `Agent.__init__(output_parser=...)` | `AgentExecutor` 内部处理 |

### 6.2 关键差异

| 方面 | 当前代码 | LangChain |
|------|---------|-----------|
| 解析失败处理 | 抛异常，携带原始文本 | 抛异常 + `OutputFixingParser` 自动修复 |
| JSON 提取 | 三重策略（直接/代码块/括号计数） | 类似逻辑 |
| Pydantic 模型 | 自动推导格式说明 | 支持 `model_json_schema()` |
| 列表解析 | 多格式智能识别 | 仅逗号分隔 |

### 6.3 我们的增强

相比 LangChain 的 `CommaSeparatedListOutputParser`，我们的实现做了增强：

- **编号列表识别**：自动识别 `1. xxx\n2. xxx` 格式
- **中文逗号兼容**：自动处理 `，` 分隔
- **JSON 数组 fallback**：输入是 JSON 数组字符串时直接解析
- **引号剥离**：自动去除元素两端的引号和括号

---

*本文档对应代码版本：2026-05-18*
