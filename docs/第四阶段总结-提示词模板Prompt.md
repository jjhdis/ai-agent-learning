# 第四阶段总结：提示词模板（Prompt Template）

> **学习时间**：2026-05-15
> **核心问题**：为什么需要多种 Prompt 模板？它们各自解决什么问题？

---

## 目录

1. [为什么需要 Prompt 模板？](#1-为什么需要-prompt-模板)
2. [四个核心组件详解](#2-四个核心组件详解)
3. [组件关系图](#3-组件关系图)
4. [LangChain 的 Prompt 管理体系](#4-langchain-的-prompt-管理体系)
5. [与 Memory 的配合：实现"调教记忆"](#5-与-memory-的配合实现调教记忆)
6. [当前代码 vs LangChain 对比](#6-当前代码-vs-langchain-对比)

---

## 1. 为什么需要 Prompt 模板？

### 1.1 当前代码的局限

```python
# 当前 Agent：prompt 是硬编码的
self.history.append({"role": "user", "content": user_message})
# 没有 system prompt，没有模板，全靠手动拼接
```

问题：
- **system prompt 写死**：换场景就得改代码
- **无法动态控制长度**：示例多了浪费 Token，少了效果差
- **历史消息没地方插**：只能手动拼到列表里
- **不可复用**：不同 Agent 要重新写一遍

### 1.2 模板化解决什么问题

| 问题 | 模板方案 |
|------|----------|
| prompt 写死 | 用 `{variable}` 占位符动态填充 |
| 示例管理 | `FewShotPromptTemplate` 自动选示例、控长度 |
| 历史插入 | `MessagePlaceholder` 留插入点 |
| 多角色对话 | `ChatPromptTemplate` 管理 system/user/assistant |
| 模块复用 | 模板可组合、可 partial、可继承 |

---

## 2. 四个核心组件详解

### 2.1 PromptTemplate —— 基础字符串模板

**文件**：`agent/prompt/base.py`

**本质**：带 `{variable}` 占位符的字符串模板引擎。

```python
tpl = PromptTemplate("你好，{name}！今天{city}天气如何？")
tpl.format(name="小明", city="上海")
# → "你好，小明！今天上海天气如何？"
```

**核心能力**：
- `format(**kwargs)`：变量替换
- `partial(**kwargs)`：预设部分变量，返回新模板
- `invoke(input_data)`：Runnable 协议接口
- 自动检测缺失变量并报错

**类比**：Python 的 `f-string` 或 `str.format()`，但更灵活（支持部分填充、可组合）。

**使用场景**：任何需要动态生成字符串的地方。

---

### 2.2 ChatPromptTemplate —— 多轮对话模板

**文件**：`agent/prompt/chat.py`

**本质**：管理一组带角色的消息模板（system/user/assistant），每条消息内部用 `PromptTemplate` 做变量替换。

```python
tpl = ChatPromptTemplate([
    ("system", "你是一个{role}助手，擅长{skill}"),
    ("user", "{input}"),
])
tpl.format_messages(role="天气", skill="查询天气", input="上海天气")
# → [
#     {"role": "system", "content": "你是一个天气助手，擅长查询天气"},
#     {"role": "user", "content": "上海天气"},
# ]
```

**核心区别**：不返回字符串，返回**消息列表**（`list[dict]`），直接给 LLM 用。

**支持**：
- `(role, template)` 元组：role 为 "system" / "user" / "assistant"
- `MessagePlaceholder`：运行时插入消息列表
- `from_messages()`：工厂方法

**类比**：`PromptTemplate` 是"一句话模板"，`ChatPromptTemplate` 是"整段对话模板"。

---

### 2.3 FewShotPromptTemplate —— 少样本示例模板

**文件**：`agent/prompt/few_shot.py`

**本质**：在提示词中动态插入示例，结构为 `前缀 + 示例列表 + 后缀`。

```python
few_shot = FewShotPromptTemplate(
    example_selector=selector,    # 示例选择器
    example_prompt=PromptTemplate("输入: {input}\n回答: {answer}"),
    prefix="以下是一些对话示例:\n",
    suffix="\n现在请回答: {input}",
)
few_shot.format(input="上海天气如何？")
# → "以下是一些对话示例:\n输入: 北京天气\n回答: 晴\n\n输入: 广州天气\n回答: 多云\n\n现在请回答: 上海天气如何？"
```

**核心组件**：

| 组件 | 作用 |
|------|------|
| `BaseExampleSelector` | 示例选择器抽象基类 |
| `LengthBasedExampleSelector` | 基于 Token 长度选示例，越新优先级越高 |
| `FewShotPromptTemplate` | 拼装前缀 + 示例 + 后缀 |

**LengthBasedExampleSelector 的工作方式**：
1. 从示例池中**从后向前**遍历（越新越相关）
2. 格式化每个示例，计算长度
3. 累加不超过 `max_tokens` 限制
4. 返回选中的示例子集

**类比**：`PromptTemplate` 是"填空"，`FewShotPromptTemplate` 是"先给几个例子，再问问题"。

---

### 2.4 MessagePlaceholder —— 消息列表占位符

**文件**：`agent/prompt/placeholder.py`

**本质**：在 `ChatPromptTemplate` 中标记一个位置，运行时插入一整段消息列表。

```python
tpl = ChatPromptTemplate([
    ("system", "你是一个AI助手"),
    MessagePlaceholder("history"),     # ← 运行时插入历史消息
    ("user", "{input}"),
])
tpl.format_messages(
    input="帮我查天气",
    history=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    ]
)
```

**核心区别**：不是做变量替换，而是**插入整个消息数组**。

**使用场景**：
- 插入对话历史（history）
- 插入多轮上下文
- 插入检索到的文档片段

**类比**：就像写作文时留一个"【此处插入图片】"的标记，运行时再把图片贴进去。

---

## 3. 组件关系图

```
PromptTemplate（基础字符串替换）
    ↑ 被用于
ChatPromptTemplate（多轮对话管理）
    ↑ 可以包含
MessagePlaceholder（消息列表插入点）

FewShotPromptTemplate（独立体系）
    ↑ 依赖
BaseExampleSelector（示例选择器）
    ↑ 实现
LengthBasedExampleSelector（按长度选示例）
```

**它们的关系**：
- `ChatPromptTemplate` 内部用 `PromptTemplate` 处理每条消息的变量替换
- `ChatPromptTemplate` 可以包含 `MessagePlaceholder` 作为占位项
- `FewShotPromptTemplate` 是独立体系，专门处理"给例子"的场景

---

## 4. LangChain 的 Prompt 管理体系

LangChain 的 Prompt 管理是**分层体系**，从简单到复杂：

### 第一层：PromptTemplate
```python
from langchain.prompts import PromptTemplate
prompt = PromptTemplate.from_template("你好，{name}！")
```

### 第二层：ChatPromptTemplate
```python
from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}助手"),
    ("user", "{input}"),
])
```

### 第三层：MessagesPlaceholder
```python
from langchain.prompts import MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
])
```

### 第四层：FewShotPromptTemplate
```python
from langchain.prompts import FewShotPromptTemplate
# 带示例选择器，动态控制示例数量和长度
```

### 第五层：PipelinePromptTemplate（你的代码还没有）
```python
from langchain.prompts.pipeline import PipelinePromptTemplate
# 把多个 PromptTemplate 组合成一个
# 例如：intro + personality + main = 完整 prompt
```

### 第六层：与 Chain 集成（你的代码还没有）
```python
# LangChain 把 prompt 作为链中的一个节点
chain = prompt | llm | output_parser
chain.invoke({"role": "天气", "input": "上海天气"})
```

**LangChain 的精髓**：不是"怎么写 prompt"，而是**"怎么组织、复用、组合 prompt"**。

---

## 5. 与 Memory 的配合：实现"调教记忆"

这是你在对话中提到的想法——把对话历史总结成 summary，每次新会话拼到 prompt 里。

### 5.1 思路

```
阶段1：调教模式
用户：你要用文言文回答
AI：诺，谨遵君命。
用户：每次回答前先引用一句古诗
AI：遵命。"床前明月光"——请问您有何吩咐？
...

阶段2：生成 summary
→ LLM 总结出："用户要求用文言文回答，且每次回答前引用一句古诗"

阶段3：正式使用
每次新对话 → system prompt 里拼上 summary
→ AI 自动保持文言文 + 古诗风格
```

### 5.2 代码实现思路

```python
class ConversationSummaryMemory(BaseMemory):
    """用 LLM 总结对话，每次只把摘要塞进 prompt"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.summary = ""
    
    def load(self) -> list[dict]:
        if not self.summary:
            return []
        return [
            {"role": "system", "content": f"以下是之前的对话总结，请保持一致的风格：\n{self.summary}"}
        ]
    
    def save(self, context: list[dict]) -> None:
        prompt = f"请总结以下对话的核心信息（人设、回答风格、关键约定）：\n{context}\n总结："
        response = self.llm.chat([{"role": "user", "content": prompt}])
        self.summary = response.choices[0].message.content
```

### 5.3 与 PromptTemplate 配合

```python
# 用模板来拼装
prompt = ChatPromptTemplate([
    ("system", "你是一个AI助手。\n【历史风格总结】\n{summary}"),
    MessagePlaceholder("history"),  # 近几轮对话
    ("user", "{input}"),
])

# 每次调用
messages = prompt.format_messages(
    summary=memory.summary,      # 从记忆拿摘要
    history=recent_history,       # 最近几轮对话
    input=user_message
)
```

这就是 `MessagePlaceholder` 的价值——它给了你一个"插入点"，让你可以把记忆（history）灵活地塞进对话模板的中间位置。

---

## 6. 当前代码 vs LangChain 对比

| 功能 | 你的代码 | LangChain |
|------|----------|-----------|
| PromptTemplate | ✅ `agent/prompt/base.py` | `langchain.prompts.PromptTemplate` |
| ChatPromptTemplate | ✅ `agent/prompt/chat.py` | `langchain.prompts.ChatPromptTemplate` |
| MessagePlaceholder | ✅ `agent/prompt/placeholder.py` | `langchain.prompts.MessagesPlaceholder` |
| FewShotPromptTemplate | ✅ `agent/prompt/few_shot.py` | `langchain.prompts.FewShotPromptTemplate` |
| 示例选择器 | ✅ `LengthBasedExampleSelector` | `langchain.prompts.example_selector` |
| PipelinePromptTemplate | ❌ 没有 | 组合多个模板 |
| 与 Chain 集成 | ❌ 没有 | `prompt \| llm \| parser` |
| Prompt Hub | ❌ 没有 | 社区共享模板库 |
| Message 对象 | ❌ 用 dict | `BaseMessage` 子类（有额外能力） |

### 关键差距

1. **PipelinePromptTemplate**：模块化组合 prompt，把大 prompt 拆成可复用的小模块
2. **与 Chain 集成**：让 prompt 成为链中的一个可插拔节点，而不是藏在 Agent 内部
3. **Message 对象**：LangChain 的 Message 有序列化、token 计数等扩展能力

---

## 总结

Prompt 模板不是"写死一遍的东西"，而是**把 prompt 的构建从 Agent 内部抽离出来**，让 prompt 变成可组合、可复用的组件。

四个组件的分工：

| 组件 | 输入 | 输出 | 核心用途 |
|------|------|------|----------|
| `PromptTemplate` | 变量键值对 | **字符串** | 单条消息的模板填充 |
| `ChatPromptTemplate` | 变量键值对 | **消息列表** | 构建完整的多轮对话 |
| `FewShotPromptTemplate` | 变量键值对 | **字符串** | 动态插入示例到提示词 |
| `MessagePlaceholder` | 消息列表 | **消息列表**（展开插入） | 在对话中插入历史/上下文 |

**核心思想**：不是"怎么写 prompt"，而是**"怎么组织、复用、组合 prompt"**。
