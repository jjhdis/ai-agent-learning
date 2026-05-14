# 第二阶段总结：对话记忆（Memory）

## 一、核心概念：self.memory vs self.history

### 代码层面

```python
class Agent:
    def __init__(self, ..., memory: BaseMemory = None):
        self.history: list[dict] = []  # 当前对话的消息列表
        self.memory = memory           # 外部存储策略
```

- **`self.history`**：始终是一个 `list[dict]`，存放当前正在使用的消息列表
- **`self.memory`**：一个封装对象，负责在多次 `run()` 调用之间保存/恢复 `self.history`

### 关键理解

`self.history` 是 Agent 的**实例属性**，只在 `__init__` 中初始化一次。即使没有 Memory，连续调用 `agent.run("上海天气")` → `agent.run("北京呢？")`，第二次调用时 `self.history` 中仍然保留着第一次的对话记录。

**Memory 的真正价值**不在于"能不能跨轮次保存"（这个 `self.history` 本身就能做到），而在于：

1. **提供不同的存储策略**（全量/窗口/摘要）
2. **封装"存/取/清"的接口**，Agent 不需要关心具体实现
3. **让策略可切换**，同一 Agent 可在不同场景用不同记忆模式

### 类比

- `self.history` = 办公桌上的文件（当前正在用的）
- `self.memory` = 智能文件柜（存什么、怎么归档的策略）
- 没有 Memory = 每次上班桌子都是空的

---

## 二、三种基础记忆模式

### 1. Buffer（完整记忆）—— `ConversationBufferMemory`

```python
class ConversationBufferMemory(BaseMemory):
    def __init__(self, max_messages: int = 0):
        self._history: list[dict] = []

    def save(self, context: list[dict]) -> None:
        self._history = list(context)  # 原样复制
        if self.max_messages > 0 and len(self._history) > self.max_messages:
            self._history = self._history[-self.max_messages:]  # 超限才裁剪

    def load(self) -> list[dict]:
        return list(self._history)
```

**行为**：原样保存全部历史，`load()` 时全部返回。

**特点**：
- ✅ 信息 100% 完整
- ❌ Token 消耗随对话轮次线性增长
- ❌ 无限增长可能超出 LLM 上下文窗口

### 2. Window（滑动窗口）—— `ConversationBufferWindowMemory`

```python
class ConversationBufferWindowMemory(BaseMemory):
    def __init__(self, k: int = 3):
        self.k = k
        self._history: list[dict] = []

    def save(self, context: list[dict]) -> None:
        self._history = list(context)
        self._trim()

    def _trim(self) -> None:
        """只保留最近 K 轮对话。"""
        user_indices = [
            i for i, m in enumerate(self._history)
            if m.get("role") == "user"
        ]
        if len(user_indices) <= self.k:
            return  # 不足 K 轮，不裁剪
        cutoff = user_indices[-(self.k)]
        self._history = self._history[cutoff:]
```

**行为**：只保留最近 K 轮对话（以 user 消息计数），更早的一刀切丢弃。

**特点**：
- ✅ Token 消耗稳定（始终是 K 轮的量）
- ✅ 实现简单，速度快
- ❌ 早期对话完全丢失，LLM 无法回忆

### 3. Summary（摘要记忆）—— `ConversationSummaryMemory`

```python
class ConversationSummaryMemory(BaseMemory):
    def __init__(self, llm: LLMClient, buffer_size: int = 6):
        self.llm = llm
        self.buffer_size = buffer_size
        self._summary: str = ""
        self._history: list[dict] = []

    def save(self, context: list[dict]) -> None:
        if len(context) > self.buffer_size:
            to_summarize = context[:-self.buffer_size]   # 旧消息 → 压缩
            self._history = context[-self.buffer_size:]  # 最近消息 → 保留
            self._summarize(to_summarize)
        else:
            self._history = list(context)

    def load(self) -> list[dict]:
        if not self._summary:
            return list(self._history)
        return [{"role": "system", "content": f"[对话历史摘要] {self._summary}"}] + list(self._history)

    def _summarize(self, messages: list[dict]) -> None:
        history_text = "\n".join(
            f"[{m['role']}]: {m.get('content', '') or '(调用工具)'}"
            for m in messages
        )
        prompt = SUMMARY_PROMPT.format(history=history_text)
        msg = self.llm.chat([{"role": "user", "content": prompt}])
        self._summary = msg.content or ""
```

**行为**：当消息数超过 `buffer_size` 时，将早期消息用 LLM 压缩成一段摘要，保留最近 `buffer_size` 条原始消息。

**特点**：
- ✅ Token 消耗稳定（摘要 + 近期消息）
- ✅ 历史信息不丢失（被压缩成摘要）
- ❌ 每次压缩需要调用 LLM（额外成本 + 延迟）
- ❌ 摘要可能丢失细节

---

## 三、三种模式对比

| 维度 | Buffer | Window (K=2) | Summary (buffer_size=4) |
|------|:------:|:------------:|:-----------------------:|
| **save() 做了什么** | 原样复制 | 找 user 索引，裁剪 | 分两段，前半段调 LLM 压缩 |
| **load() 返回什么** | 全部原始消息 | 最近 K 轮的原始消息 | 1 条摘要 + 最近 buffer_size 条原始消息 |
| **历史怎么丢的** | 不丢（除非设 max_messages） | 一刀切丢掉前面的 | 旧消息被压缩成摘要，不丢信息但变短了 |
| **Token 消耗** | 越来越高 | 稳定（K 轮的量） | 稳定（摘要 + buffer_size 条） |
| **信息完整度** | 100% | 只记得最近 K 轮 | 历史摘要 + 近期细节 |
| **额外成本** | 无 | 无 | 每次压缩要调一次 LLM |

### 为什么演示中三种模式输出看起来一样？

因为演示只有两轮对话：

- **Window (K=2)**：2 轮对话正好有 2 条 user 消息，`len(user_indices) <= k` 条件成立，**不触发裁剪**
- **Summary (buffer_size=4)**：虽然触发了压缩，但概览打印被截断，看起来和 Buffer 前几条差不多

**需要 5 轮以上对话才能看到明显差异**。

---

## 四、生产环境：三层记忆架构

### LangChain 的真实策略

LangChain 提供了基础记忆类型，但生产环境通常**组合使用**：

```
短期记忆 (Buffer/Window)
  └─ 最近 N 条原始消息，完整保留
  └─ 类似 Window 或带上限的 Buffer

中期记忆 (Summary)
  └─ 对早期对话的压缩摘要
  └─ 类似 Summary，但按 token 数动态触发（而非固定消息条数）

长期记忆 (VectorStore)
  └─ 所有历史对话的向量化存储
  └─ 每次对话检索最相关的 3-5 条历史
```

### 推荐方案：ConversationSummaryBufferMemory

LangChain 最推荐的方案，**结合了 Summary 和 Buffer 的优点**：

```
工作方式：
- 设定 max_token_limit（如 2000 tokens）
- 总 token 没超限 → 全部保留原始消息（像 Buffer）
- 一旦超限 → 最早的消息压缩成摘要（像 Summary）
- 下次再超限 → 旧摘要 + 新一批消息再压缩
```

### VectorStore 记忆

VectorStore（向量数据库）本身只有两个能力：
1. **存**：文本 → 向量（embedding）→ 存入数据库
2. **取**：查询文本 → 转向量 → 语义相似度搜索 → 返回最相关历史

它**没有摘要能力**，但可以和 Summary 组合使用：
- 存原始消息 → 检索到原文（信息无损但占 token）
- 存摘要 → 检索到摘要（节省 token 但可能丢细节）

### 真正的"切换"不是换 Memory，而是配置参数

```python
memory = HierarchicalMemory(
    llm=llm,
    window_k=5,           # 短期保留几轮
    summary_buffer=10,    # 多少条消息后触发摘要
    summary_llm=llm,      # 摘要用的模型（可用更便宜的）
    vectorstore=chroma,   # 长期向量库
    retrieval_k=3,        # 每次检索几条相关历史
)
```

---

## 五、关键知识点

1. **`self.history` 是实例属性**，即使没有 Memory 也能跨 `run()` 保留
2. **Memory 是封装 + 多态**：基类定义接口（load/save/clear），子类实现不同策略
3. **Window 按 user 消息计数轮次**，不是按消息条数
4. **Summary 需要 LLM 参与**，有额外成本和延迟
5. **生产环境用组合架构**，不是单一策略
6. **VectorStore 本身无摘要能力**，需要和 Summary 组合使用

---

## 六、后续学习方向

1. 实现 `ConversationSummaryBufferMemory`（按 token 数动态压缩）
2. 实现 `HierarchicalMemory`（三层组合记忆）
3. 引入向量数据库（Chroma/FAISS）做长期记忆
4. 了解专用记忆服务（Zep / Mem0）
