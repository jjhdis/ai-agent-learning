# 第一阶段总结：链式组合（Chain）

> **对应学习路线**：第一阶段 — 链式组合（Chain）
> **核心代码**：`agent/chain/runnable.py`、`agent/chain/passthrough.py`、`main.py` 中的 `demo_travel_chain()`
> **日期**：2026-05-13

---

## 目录

1. [我们做了什么](#1-我们做了什么)
2. [核心概念：管道（Pipeline/Chain）](#2-核心概念管道pipelinechain)
3. [代码实现详解](#3-代码实现详解)
4. [运行效果](#4-运行效果)
5. [你问过的问题与解答](#5-你问过的问题与解答)
6. [管道的真正用途](#6-管道的真正用途)
7. [下一步学习方向](#7-下一步学习方向)

---

## 1. 我们做了什么

在第一阶段，我们在已有 Agent 代码的基础上，实现了一个 **Chain 管道系统**，并用一个"十一假期出行推荐"的 Demo 来演示它的工作原理。

### 已实现的组件

| 组件 | 文件 | 作用 |
|------|------|------|
| `Runnable`（抽象基类） | `agent/chain/runnable.py` | 定义统一的 `invoke()` 接口 |
| `RunnableLambda` | `agent/chain/passthrough.py` | 把普通函数包装成 Runnable |
| `RunnablePassthrough` | `agent/chain/passthrough.py` | 原样透传输入数据 |
| `RunnableMap` | `agent/chain/passthrough.py` | 并行执行多个 Runnable，合并输出为 dict |
| `\|` 管道操作符 | `agent/chain/runnable.py` | 把多个 Runnable 串联成管道 |

### Demo 演示的管道

```
用户输入："十一假期我想出去玩，有什么推荐？"
    │
    ▼
步骤1: analyze_preference  ──→  输出: {"preference": "自然风光", ...}
(RunnableLambda)                   (dict)
    │
    ▼
步骤2: recommend_cities  ──────→  输出: {"preferences": {...}, "cities": ["稻城亚丁", "桂林", "张家界"]}
(RunnableLambda)                   (dict)
    │
    ▼
步骤3: check_cities_weather  ──→  输出: {"preferences": {...}, "cities": [...], "weathers": {...}}
(RunnableLambda)                   (dict，内部用 RunnableMap 并行查天气)
    │
    ▼
步骤4: make_recommendation  ──→  输出: "推荐桂林和张家界..."
(RunnableLambda)                   (最终文本)
```

---

## 2. 核心概念：管道（Pipeline/Chain）

### 2.1 什么是管道？

**管道（Pipeline/Chain）** 就是把多个处理步骤串联起来，让数据像水流过管道一样，**前一步的输出自动成为后一步的输入**。

```
输入 → [步骤1] → [步骤2] → [步骤3] → 输出
```

### 2.2 生活中的类比

**工厂流水线**：
```
原料 → [切割] → [打磨] → [组装] → [质检] → 成品
```

每个工位只做一件事，做完传给下一个工位。原料经过所有工位后，就变成了成品。

### 2.3 代码中的体现

在我们的代码中，管道是这样组装的：

```python
travel_chain = (
    RunnableLambda(analyze_preference)      # 步骤1
    | RunnableLambda(recommend_cities)       # 步骤2
    | RunnableLambda(check_cities_weather)   # 步骤3
    | RunnableLambda(make_recommendation)    # 步骤4
)

result = travel_chain.invoke("十一假期我想出去玩")
```

**关键点**：
- `|` 符号把四个 Runnable **串联**起来
- 调用 `travel_chain.invoke()` 时，数据依次流过每个步骤
- **前一步的返回值，自动传给后一步作为参数**

### 2.4 数据流追踪

为了让你更直观地理解"前一步的输出是后一步的输入"，我们追踪一下数据的变化：

```
步骤1 输出: {"preference": "自然风光", "budget": "中等", "with_who": "朋友"}
    ↓ 这个 dict 自动传给步骤2
步骤2 接收: {"preference": "自然风光", "budget": "中等", "with_who": "朋友"}
步骤2 输出: {"preferences": {...}, "cities": ["稻城亚丁", "桂林", "张家界"]}
    ↓ 这个 dict 自动传给步骤3
步骤3 接收: {"preferences": {...}, "cities": ["稻城亚丁", "桂林", "张家界"]}
步骤3 输出: {"preferences": {...}, "cities": [...], "weathers": {"稻城亚丁": "...", "桂林": "...", "张家界": "..."}}
    ↓ 这个 dict 自动传给步骤4
步骤4 接收: {"preferences": {...}, "cities": [...], "weathers": {...}}
步骤4 输出: "推荐桂林和张家界..."
```

**每一步的输出 dict 都在"长大"**，后面的步骤可以访问前面所有步骤产生的结果。

---

## 3. 代码实现详解

### 3.1 组件一览

| 组件 | 文件 | 一句话概括 | 核心代码 |
|------|------|-----------|---------|
| **Runnable** (基类) | `runnable.py` | 定义统一接口，所有组件都继承它 | `invoke(input) → output` + `__or__` |
| **RunnableLambda** | `passthrough.py` | 把普通函数包装成 Runnable | `invoke(x) = func(x)` |
| **RunnableMap** | `passthrough.py` | 并行执行多个 Runnable，合并为 dict | `invoke(x) = {k: r.invoke(x)}` |
| **RunnableSequence** | `runnable.py` | 串联多个 Runnable，前一步输出=后一步输入 | `for step in steps: result = step.invoke(result)` |
| **RunnablePassthrough** | `passthrough.py` | 原样透传，不做任何处理 | `invoke(x) = x` |

### 3.2 核心机制：`|` 操作符（管道符号）

`|` 之所以能自动传参，靠的是 Python 的 **`__or__` 魔术方法**。写 `a | b` 等价于调用 `a.__or__(b)`。

#### ① `__or__` 做了什么？

```python
def __or__(self, other):
    steps = []
    # 如果左边已经是 RunnableSequence，展开它（避免嵌套）
    if isinstance(self, RunnableSequence):
        steps.extend(self.steps)
    else:
        steps.append(self)
    # 右边同理
    if isinstance(other, RunnableSequence):
        steps.extend(other.steps)
    else:
        steps.append(other)
    return RunnableSequence(steps)  # 返回一个"步骤列表"
```

**关键设计**：检查左右两边是否已经是 `RunnableSequence`，如果是则**展开**再合并，避免产生"套娃"。

#### ② `a | b | c | d` 的组装过程（从左到右逐步合并）

```
第1步: a | b           → RunnableSequence([a, b])
第2步: [a,b] | c       → RunnableSequence([a, b, c])
第3步: [a,b,c] | d     → RunnableSequence([a, b, c, d])
```

> ⚠️ **重要理解**：`|` 只是在**组装**管道，不是在**执行**。`travel_chain = a | b | c | d` 只是创建了一个存有 4 个步骤的 `RunnableSequence` 对象，**还没有执行任何一步**。

#### ③ 真正执行时：`invoke()` 依次调用

```python
def invoke(self, input_data):
    result = input_data
    for step in self.steps:          # 遍历步骤列表
        result = step.invoke(result) # ★ 上一步的输出 = 下一步的输入
    return result
```

**执行追踪**（以 `travel_chain.invoke("十一假期我想出去玩")` 为例）：

| 轮次 | 执行的步骤 | 输入 | 输出 |
|------|-----------|------|------|
| 1 | `analyze_preference` | `"十一假期我想出去玩"` | `{"preference": "自然风光", ...}` |
| 2 | `recommend_cities` | ↑ 上一步的 dict | `{"preferences": {...}, "cities": [...]}` |
| 3 | `check_cities_weather` | ↑ 上一步的 dict | `{"preferences": {...}, "cities": [...], "weathers": {...}}` |
| 4 | `make_recommendation` | ↑ 上一步的 dict | `"推荐桂林和张家界..."` |

**每一步的输出 dict 都在"长大"**，后面的步骤可以访问前面所有步骤产生的结果。

### 3.3 生活类比：工厂生产线

```
         | 符号 = 传送带（连接工位）
         
  [分析偏好] ----→ [推荐城市] ----→ [查天气] ----→ [综合推荐]
     工位1           工位2           工位3           工位4
```

- **`travel_chain = a \| b \| c \| d`** = 搭生产线，把工位用传送带连起来，机器还没开动
- **`travel_chain.invoke(input)`** = 按下启动按钮，原料从工位1开始依次加工
- **每个工位的输出自动掉到传送带上**，送到下一个工位的入口

### 3.4 对比：不用 `|` 的手动写法

```python
# ❌ 手动串联（繁琐、难维护）
result1 = analyze_preference("十一假期我想出去玩")
result2 = recommend_cities(result1)
result3 = check_cities_weather(result2)
result4 = make_recommendation(result3)

# ✅ 用 | 串联（简洁、可复用）
travel_chain = (
    RunnableLambda(analyze_preference)
    | RunnableLambda(recommend_cities)
    | RunnableLambda(check_cities_weather)
    | RunnableLambda(make_recommendation)
)
result = travel_chain.invoke("十一假期我想出去玩")
```

**管道的优势**：
1. **可复用**：定义一次 `travel_chain`，可以多次调用 `.invoke()` 传入不同输入
2. **可组合**：可以随时在中间插入/删除步骤，不需要改调用代码
3. **可观察**：可以给 `RunnableSequence` 加日志，追踪每一步的输入输出
---

## 4. 运行效果

执行 `python main.py` 后输入 `/demo`，可以看到完整的管道执行过程：

```
【管道结构】
  步骤1: 分析偏好  →  步骤2: 推荐城市  →  步骤3: 查天气  →  步骤4: 综合推荐
  (RunnableLambda)  (RunnableLambda)   (RunnableMap)    (RunnableLambda)

[用户输入]: 十一假期我想出去玩，有什么推荐？

[Agent] 分析偏好 → {"preference": "自然风光", "budget": "中等", "with_who": "朋友"}
[Agent] 推荐城市 → {"cities": ["稻城亚丁", "桂林", "张家界"]}
[Agent] 并行查天气 → {"稻城亚丁": "...", "桂林": "...", "张家界": "..."}
[Agent] 综合推荐 → "推荐桂林和张家界..."

[最终推荐]
## 推荐城市：桂林、张家界
...
```

---

## 5. 你问过的问题与解答

### Q1: 什么叫"前一步的输出是后一步的输入"？代码中怎么体现？

**答**：看 `RunnableSequence.invoke()` 的核心代码：

```python
result = input_data
for step in self.steps:
    result = step.invoke(result)  # 把上一步的结果传给下一步
return result
```

具体到我们的 Demo：

```python
# 步骤1 的输出
result1 = analyze_preference.invoke("十一假期我想出去玩")
# result1 = {"preference": "自然风光", ...}

# 步骤1 的输出 result1 自动传给步骤2
result2 = recommend_cities.invoke(result1)
# result2 = {"preferences": {...}, "cities": [...]}

# 步骤2 的输出 result2 自动传给步骤3
result3 = check_cities_weather.invoke(result2)
# result3 = {"preferences": {...}, "cities": [...], "weathers": {...}}

# 步骤3 的输出 result3 自动传给步骤4
result4 = make_recommendation.invoke(result3)
# result4 = "推荐桂林和张家界..."
```

**简单说**：每个函数的返回值，自动成为下一个函数的参数。你不需要手动传递数据，管道帮你做了。

### Q2: 这个例子太模板化了，实际中管道到底有什么用？

**答**：管道的真正用途是**编排确定性流程**，而不是让 AI 自由发挥。典型场景：

| 场景 | 管道步骤 | 说明 |
|------|---------|------|
| **RAG（检索增强生成）** | 检索 → 组装 prompt → 调 LLM → 格式化输出 | 每次流程固定，但检索内容不同 |
| **数据处理流水线** | 清洗 → 并行处理（翻译+提取+分析）→ 合并 | 步骤固定，数据不同 |
| **多模型协作** | 分类 → 路由到不同模型 → 统一格式 | 路由逻辑固定 |
| **内容审核** | 敏感词检测 → 质量评分 → 格式化输出 | 审核流程固定 |

**管道 vs Agent 的区别**：

| | 管道（Chain） | Agent |
|---|---|---|
| **谁决定步骤** | 人（代码写死） | AI（自主决策） |
| **适用场景** | 流程确定的场景 | 不确定的复杂场景 |
| **可预测性** | 高（每一步做什么是固定的） | 低（AI 可能走不同路径） |
| **典型用途** | RAG、数据处理、多步编排 | 客服、复杂问答、多工具组合 |

### Q3: 为什么 LLM 返回的 JSON 格式不一致？

**答**：LLM 不是程序，它返回的内容可能有变化。比如我们要求返回 `["成都", "西安", "大理"]`，但 LLM 可能返回 `{"cities": ["成都", "西安", "大理"]}`。

我们的代码中做了兼容处理：

```python
parsed = json.loads(result)
if isinstance(parsed, list):
    cities = parsed  # 直接是列表
elif isinstance(parsed, dict):
    # 从 dict 中提取城市列表
    for key in ("cities", "city", "recommendations"):
        if key in parsed and isinstance(parsed[key], list):
            cities = parsed[key]
            break
```

这就是为什么实际项目中会用 **输出解析器（Output Parser）** 来规范化 LLM 的输出。

### Q4: RunnableMap 的闭包陷阱是什么？

**答**：在循环中创建 lambda 时，如果直接使用循环变量，所有 lambda 会共享同一个变量值（最后一个值）。

```python
# ❌ 错误写法：所有 lambda 的 c 都是最后一个 city 的值
for city in cities:
    weather_checks[city] = RunnableLambda(lambda c=city: ...)

# ✅ 正确写法：用函数工厂创建独立作用域
def make_weather_runnable(city_name):
    return RunnableLambda(lambda _: mock_weather.get(city_name, ...))

weather_checks = {city: make_weather_runnable(city) for city in cities}
```

---

## 6. 管道的真正用途

### 6.1 一句话总结

> **管道 = 把确定性的处理步骤串起来，让数据像流水一样流过每个工位。**

### 6.2 什么时候用管道？

**用管道**（Chain）：
- 流程是确定的，你知道每一步要做什么
- 需要保证执行顺序
- 需要并行处理多个独立任务
- 需要把多个组件组合成一个可复用的单元

**用 Agent**：
- 流程不确定，需要 AI 自己决定下一步
- 需要 AI 自主选择工具和参数
- 需要处理开放式的复杂问题

### 6.3 实际项目中的组合使用

在实际项目中，**管道和 Agent 经常组合使用**：

```
用户输入
    │
    ▼
[RunnableLambda: 意图分类]  →  判断是"查询"还是"操作"
    │
    ├── "查询" → [Agent: 自主查数据并回答]
    │
    └── "操作" → [管道: 固定流程执行操作]
                    ├── 验证权限
                    ├── 执行操作
                    └── 返回结果
```

---

## 7. 下一步学习方向

第一阶段完成后，接下来的学习路线：

| 阶段 | 知识点 | 说明 |
|------|--------|------|
| **第二阶段** | **记忆系统（Memory）** | 让 Agent 记住多轮对话，支持上下文理解 |
| **第三阶段** | **回调系统（Callback）** | 更丰富的事件钩子，支持日志、追踪 |
| **第四阶段** | **提示词模板（Prompt Template）** | 模板化管理 prompt，支持变量替换 |
| **第五阶段** | **输出解析（Output Parsing）** | 规范化 LLM 输出，支持结构化数据 |
| **第六阶段** | **流式输出（Streaming）** | 逐 Token 输出，提升用户体验 |
| **第七阶段** | **RAG（检索增强生成）** | 从外部知识库检索信息 |
| **第八阶段** | **高级 Agent 类型** | Plan-and-Execute、Self-Ask 等 |
| **第九阶段** | **生产化** | 错误重试、速率限制、监控等 |

---

*本文档对应代码版本：2026-05-13*
