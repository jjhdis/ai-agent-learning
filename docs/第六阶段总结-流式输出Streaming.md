# 第六阶段总结：流式输出（Streaming）

> **学习时间**：2026-05-18
> **核心问题**：LLM 的一次性返回让用户干等数秒，如何实现逐 Token 实时输出？

---

## 目录

1. [为什么需要流式输出？](#1-为什么需要流式输出)
2. [三个核心组件详解](#2-三个核心组件详解)
3. [组件关系图](#3-组件关系图)
4. [LangChain 的流式输出体系](#4-langchain-的流式输出体系)
5. [与 ReAct 循环的配合：工具调用中的流式](#5-与-react-循环的配合工具调用中的流式)
6. [当前代码 vs LangChain 对比](#6-当前代码-vs-langchain-对比)

---

## 1. 为什么需要流式输出？

### 1.1 当前代码的局限

```python
# 当前 Agent.run()：等 LLM 全部生成完才返回
response = self.llm.chat(history, tools)   # 阻塞 2-5 秒
msg = response.choices[0].message
return msg.content  # 用户在这之前什么都看不到
```

问题：
- **用户体验差**：用户盯着空白屏幕等待 2-5 秒
- **首字延迟高**：Time-To-First-Token 完全无法感知
- **无法中止**：不能在想中途取消生成
- **无法管道化**：下游必须等全部完成才能开始处理

### 1.2 流式输出解决什么问题

| 问题 | 流式方案 |
|------|----------|
| 长时间等待 | `stream_chat()` 逐 token 产出，边生成边显示 |
| 无法感知进度 | 用户实时看到文字出现，心理等待感大幅降低 |
| 无法中途取消 | 消费者可在任意时刻 `break` 停止迭代 |
| 无法增量处理 | `StreamAccumulator.new_tokens()` 提供增量文本 |

---

## 2. 三个核心组件详解

### 2.1 LLMClient.stream_chat() —— 流式 LLM 调用

**文件**：`agent/llm/client.py`

**本质**：在 `chat()` 的基础上增加 `stream=True`，返回生成器而非完整响应。

```python
# 非流式：阻塞等待完整响应
response = llm.chat(messages, tools)     # → ChatCompletion
content = response.choices[0].message.content

# 流式：逐个产出 delta 块
for chunk in llm.stream_chat(messages, tools):
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)  # 逐字显示
```

**核心差异**：

| 特性 | `chat()` | `stream_chat()` |
|------|---------|----------------|
| 返回类型 | `ChatCompletion` | `Generator[ChatCompletionChunk]` |
| 首字延迟 | 等待全部完成 | 即时返回 |
| Usage 信息 | 直接可用 `.usage` | 需 `stream_options={"include_usage": True}` |
| 工具调用 | 完整 `tool_calls` 列表 | 跨 chunk 分片到达 |

---

### 2.2 StreamAccumulator —— 流式块累加器

**文件**：`agent/streaming/event.py`

**本质**：解决流式 API 中工具调用"分片到达"的核心难题。将 OpenAI 的 `ChatCompletionChunk` 流重组为完整消息。

**为什么需要它**：

OpenAI 流式 API 中，一次 `tool_calls` 的完整参数会跨多个 chunk 发送：

```
chunk_1: delta.tool_calls = [{"index": 0, "id": "call_abc",
                                "function": {"name": "get_wea", "arguments": ""}}]
chunk_2: delta.tool_calls = [{"index": 0,
                                "function": {"name": "ther", "arguments": "{\"city\""}}]
chunk_3: delta.tool_calls = [{"index": 0,
                                "function": {"arguments": ": \"上海\"}"}}]
```

需要按 `index` 合并 name 和 arguments 字符串，才能得到完整的工具调用。

**核心 API**：

```python
acc = StreamAccumulator()

# 消费流式块
for chunk in llm.stream_chat(messages, tools):
    acc.add_chunk(chunk)
    new_text = acc.new_tokens()        # 增量获取文本（避免重复处理）
    if new_text:
        print(new_text, end="", flush=True)

# 流结束后构建完整消息
if acc.has_tool_calls:
    msg = acc.build_message()
    # → {"role": "assistant", "tool_calls": [
    #       {"id": "call_abc", "function": {"name": "get_weather",
    #        "arguments": "{\"city\": \"上海\"}"}}
    #    ]}
else:
    reply = acc.content  # → str 完整回复文本
```

**关键设计**：

- `new_tokens()` 返回自上次调用以来的增量文本，内部维护指针避免重复
- `build_message()` 只能在 `is_done` 为 True 时调用
- `reset()` 支持复用同一个累加器实例

---

### 2.3 StreamEvent —— 流式事件模型

**文件**：`agent/streaming/event.py`

**本质**：统一的流式事件数据结构，`Agent.run_stream()` 统一通过它产出各类运行时信息。

**事件类型**：

| 事件 | 说明 | data 内容 |
|------|------|----------|
| `THINK` | LLM 推理过程中的文本 token | `str` — 单个或少数几个字符 |
| `REPLY` | 最终回复的文本 token | `str` — 单个字符 |
| `TOOL_START` | 工具开始执行 | `{"name": str, "args": dict}` |
| `TOOL_END` | 工具执行完成 | `{"name": str, "result": str}` |
| `TOOL_ERROR` | 工具执行出错 | `{"name": str, "error": str}` |
| `DONE` | 流式输出全部完成 | 解析后的最终结果 |
| `ERROR` | 发生错误 | `str` — 错误信息 |

**设计优势**：

```python
# 消费者可以根据事件类型灵活处理
for event in agent.run_stream("上海天气"):
    if event.event == StreamEventType.REPLY:
        print(event.data, end="", flush=True)  # 实时显示
    elif event.event == StreamEventType.TOOL_START:
        print(f"\n🔧 正在查询 {event.data['name']}...")
    elif event.event == StreamEventType.DONE:
        save_to_db(event.data)  # 持久化最终结果
```

---

### 2.4 StreamingCallbackHandler —— 流式回调处理器

**文件**：`agent/streaming/handler.py`

**本质**：将 `StreamEvent` 流转换为可视化的终端输出或静默收集。

**三种模式**：

```python
# 实时模式：逐字打印（终端交互）
handler = StreamingCallbackHandler(mode="realtime")
for event in agent.run_stream("..."):
    handler.handle_event(event)

# 静默模式：只收集统计（后台处理）
handler = StreamingCallbackHandler(mode="silent")

# 详细模式：打印所有事件（调试）
handler = StreamingCallbackHandler(mode="verbose", show_thinking=True)
```

---

## 3. 组件关系图

```
Agent.run_stream(user_message)
│
├─ ReAct Loop (每轮推理)
│   │
│   ├─ LLMClient.stream_chat(history, tools)
│   │   │
│   │   └─ StreamAccumulator
│   │       ├─ .add_chunk(chunk)    ← 逐个处理 delta 块
│   │       ├─ .new_tokens()        → yield THINK event
│   │       ├─ .has_tool_calls      → 判断是否需执行工具
│   │       └─ .build_message()     → 构造完整 assistant 消息
│   │
│   ├─ [如有工具调用]
│   │   ├─ 执行工具 → yield TOOL_START / TOOL_END event
│   │   └─ 追加 tool result 到 history
│   │
│   └─ [无工具调用 — 最终回复]
│       ├─ 逐字产出 → yield REPLY event
│       ├─ 输出解析 → self.output_parser.parse(content)
│       └─ yield DONE event (携带解析后数据)
│
└─ StreamingCallbackHandler (外部消费者)
    ├─ mode="realtime"  → 实时逐字打印
    ├─ mode="verbose"   → 详细事件日志
    └─ mode="silent"    → 静默统计收集
```

---

## 4. LangChain 的流式输出体系

### 4.1 LangChain 的流式设计

LangChain 的所有 Runnable 组件都实现了 `stream()` 方法：

```python
# LangChain 流式
for chunk in chain.stream({"input": "上海天气"}):
    print(chunk, end="", flush=True)
```

每个组件的 `stream()` 内部调用下游的 `stream()`，形成完整的流式管道。

### 4.2 我们的流式设计

```python
# 我们的流式
for event in agent.run_stream("上海天气"):
    if event.event == StreamEventType.REPLY:
        print(event.data, end="", flush=True)
```

**关键差异**：

| 方面 | LangChain | 我们的实现 |
|------|-----------|----------|
| 流式粒度 | 每个组件独立 `stream()` | 统一通过 `StreamEvent` 产出 |
| 工具调用 | 内部静默处理 | 产出 `TOOL_START/END` 事件 |
| 事件类型 | 无显式类型（纯数据） | `StreamEventType` 常量 + 元数据 |
| 流式累加 | 框架内部处理 | `StreamAccumulator` 显式管理 |

### 4.3 学习收获

- **生成器（yield）是基础**：`run_stream()` 使用 `yield` 逐事件产出，消费者通过 `for` 循环消费
- **流式 + 工具调用 = 复杂度翻倍**：工具调用参数跨 chunk 分片是流式 API 最大的坑，`StreamAccumulator` 按 index 合并解决此问题
- **事件模型是解耦关键**：`StreamEventType` 将"发生了什么"和"数据是什么"分离，消费者可以按类型灵活处理

---

## 5. 与 ReAct 循环的配合：工具调用中的流式

### 5.1 挑战

ReAct 循环可能有多轮 LLM 调用：

```
用户: 上海天气怎么样？
→ LLM 调用1 (streaming): 决定调用 get_weather("上海")
→ 执行工具
→ LLM 调用2 (streaming): 给出最终回复
```

每轮 LLM 调用都是流式的，但工具执行必须等待该轮流式完全结束后才能开始。

### 5.2 解决方案

`run_stream()` 在每轮循环中：

1. 流式接收 LLM 输出，`StreamAccumulator` 累积
2. 检查 `accumulator.has_tool_calls`
3. 如有工具调用 → 执行 → 进入下一轮
4. 如无工具调用 → 这是最终回复 → 逐字产出 REPLY → 产出 DONE

```python
for step in range(self.max_iterations):
    accumulator = StreamAccumulator()
    for chunk in self.llm.stream_chat(history, tools):
        accumulator.add_chunk(chunk)
        new_text = accumulator.new_tokens()
        if new_text:
            yield StreamEvent(THINK, data=new_text)

    if not accumulator.has_tool_calls:
        # 最终回复流式输出
        for char in accumulator.content:
            yield StreamEvent(REPLY, data=char)
        yield StreamEvent(DONE, data=parsed_result)
        return

    # 构建完整消息并执行工具调用
    msg = accumulator.build_message()
    history.append(msg)
    # ... 执行工具、产出 TOOL_START/END 事件
```

---

## 6. 当前代码 vs LangChain 对比

### 6.1 功能对比

| 功能 | 当前代码 | LangChain |
|------|---------|-----------|
| 流式 LLM 调用 | `LLMClient.stream_chat()` (生成器) | `BaseChatModel.stream()` (生成器) |
| 块累加器 | `StreamAccumulator` | 框架内部处理 |
| 流式事件 | `StreamEvent` + `StreamEventType` | 直接 `yield` 数据（无类型包装） |
| 流式回调 | `StreamingCallbackHandler` | `BaseCallbackHandler.on_llm_new_token()` |
| ReAct 流式 | `Agent.run_stream()` | `AgentExecutor.stream()` |
| 工具调用流式 | 工具调用期间产出事件 | 工具调用期间静默 |
| 思考过程展示 | `StreamEventType.THINK` + `show_thinking` | 取决于具体实现 |

### 6.2 我们的特色增强

- **显式事件类型**：7 种 `StreamEventType` 常量，消费者无需猜测事件含义
- **Step 追踪**：每个事件携带 `step` 字段，标识当前 ReAct 循环轮次
- **三种回调模式**：`realtime` / `verbose` / `silent` 覆盖不同使用场景
- **流式 + 解析一体**：`DONE` 事件携带经 `output_parser` 解析后的最终数据
- **思考过程分离**：THINK vs REPLY 事件类型，支持独立控制是否显示推理过程

---

---

## 7. 常见问题与深入理解

> 以下内容来自学习过程中的提问与思考，记录了流式输出设计中容易混淆的关键问题。

### 7.1 run_stream() 中"真流式"vs"模拟流式"的矛盾

**问题**：`run_stream()` 先用 `StreamAccumulator` 把流式 chunk 全部累积完，再逐字 yield 出去，这看起来是先收集完再模拟流式，很矛盾。

**解答**：`run_stream()` 实际上包含**两个阶段**：

| 阶段 | 代码位置 | 是否真流式 | 事件类型 | 说明 |
|------|---------|-----------|---------|------|
| 阶段一 | 第 206-217 行 | ✅ **真流式** | THINK | `self.llm.stream_chat()` 返回生成器，每次 `for` 循环拿到一个 chunk，`accumulator.new_tokens()` 返回增量文本，立即 `yield` 出去 |
| 阶段二 | 第 241-248 行 | ❌ 模拟流式 | REPLY | `accumulator.content` 已经是阶段一累积完成的完整文本，拆成单字符逐个 `yield` |

**为什么要有阶段二（模拟流式）？**

因为在阶段一流式结束前，代码**无法区分**这次 LLM 调用是"工具调用"还是"最终回复"——只有等 `finish_reason` 出现才能判断。所以设计上：

1. 阶段一：**先全部流式接收并累积**，同时实时产出 THINK 事件让用户感知"正在生成"
2. 阶段一结束后：检查 `accumulator.has_tool_calls` 判断是工具调用还是最终回复
3. 如果是最终回复：**重新以 REPLY 事件类型逐字输出**，让消费者可以区分"思考过程"和"最终回复"

**如果你想要完全真正的流式**（不区分 THINK/REPLY，直接实时输出），可以这样消费：

```python
for event in agent.run_stream("上海天气"):
    if event.event in (StreamEventType.THINK, StreamEventType.REPLY):
        print(event.data, end="", flush=True)
```

这样 THINK 阶段已经实时打印了，REPLY 阶段虽然内容重复但因为是逐字输出，用户感知上仍然是流畅的。

---

### 7.2 StreamEvent 存在的意义 —— 为什么不用简单的 list/元组？

**问题**：StreamEvent 这个类设计的意义是什么？不能简单粗暴地放到 list 里吗？

**解答**：用 list + 元组确实也能实现同样的功能：

```python
# 用 list 的版本（也能工作）
events = []
events.append(("think", "上"))
events.append(("think", "海"))
events.append(("tool_start", {"name": "get_weather", "args": {"city": "上海"}}))
events.append(("done", "上海晴，28°C"))

for event_type, data in events:
    if event_type == "reply":
        print(data, end="")
```

但 StreamEvent 的设计有 3 个核心优势：

**① 流式 vs 批处理的本质区别**

```python
# list 方式：必须等所有事件都生成完才能消费
events = []
for chunk in llm.stream_chat(...):
    events.append(("think", chunk.content))
# 等全部结束才能开始消费
for event_type, data in events:
    ...

# StreamEvent 方式：来一个处理一个，不用等全部结束
for event in agent.run_stream(...):
    if event.event == StreamEventType.REPLY:
        print(event.data, end="", flush=True)  # 实时显示！
```

**② 可扩展性**

```python
# 元组方式：扩展困难
("think", "上")           # 只有类型和数据
# 想加 step 信息？得改成 ("think", "上", 0)
# 想加 metadata？得改成 ("think", "上", 0, {})
# 所有消费者都得跟着改

# StreamEvent 方式：加字段不影响已有代码
StreamEvent(event="think", data="上", step=0)
# 加 metadata 字段：已有代码完全不用改
StreamEvent(event="think", data="上", step=0, metadata={"model": "deepseek"})
```

**③ 类型安全 + IDE 支持**

```python
# 元组：IDE 不知道里面是什么
for event_type, data in events:
    # IDE 不知道 event_type 是字符串，data 是 Any
    # 拼写错误 "thnik" 也不会报错

# StreamEvent：IDE 能自动补全
for event in agent.run_stream(...):
    # 输入 event.  IDE 自动提示 event / data / step / metadata
    # 输入 StreamEventType. IDE 自动提示 THINK / REPLY / TOOL_START ...
    if event.event == StreamEventType.REPLY:
        print(event.data, end="")
```

| 对比维度 | 用 list + 元组 | 用 StreamEvent |
|---------|--------------|---------------|
| 能否工作 | ✅ 能 | ✅ 能 |
| 流式实时性 | ❌ 必须等全部收集完 | ✅ 边生成边消费 |
| 扩展性 | ❌ 加字段要改所有消费者 | ✅ 加字段不影响已有代码 |
| 可读性 | ❌ `event[0]`、`event[1]` 含义模糊 | ✅ `event.event`、`event.data` 一目了然 |
| IDE 支持 | ❌ 无自动补全 | ✅ 完整类型提示 |
| 防拼写错误 | ❌ `"thnik"` 不会报错 | ✅ `StreamEventType.THINK` 拼错 IDE 直接报红 |

**总结**：list + 元组是"能用"，StreamEvent 是"好用"。用有意义的类型包装数据，让代码自文档化。

---

### 7.3 yield vs return 的区别

**问题**：`yield` 是什么？和 `return` 有啥区别？

**解答**：核心区别用一个比喻就能说清楚：

- **`return`** = 一次性全部给你（函数结束，局部变量全部销毁）
- **`yield`** = 来一点给一点，边做边给（函数暂停，局部变量保留，下次继续）

```python
# return 版本：等全部生成完才返回
def run_stream_return(user_message):
    full_text = llm.chat(...)  # 等 3 秒
    return full_text  # 用户干等 3 秒，突然看到全部文字

result = run_stream_return("上海天气")
print(result)  # 3 秒后一次性显示全部
```

```python
# yield 版本：边生成边产出
def run_stream_yield(user_message):
    for chunk in llm.stream_chat(...):  # 逐块到达
        yield StreamEvent(REPLY, data=chunk.content)  # 来一块给一块

for event in run_stream_yield("上海天气"):
    print(event.data, end="", flush=True)  # 实时显示，不用等
```

| 特性 | `return` | `yield` |
|------|---------|---------|
| 函数状态 | 函数结束，局部变量全部销毁 | 函数暂停，局部变量保留 |
| 调用结果 | 返回具体的值 | 返回生成器对象 |
| 执行方式 | 一次性执行完 | 可中断/可恢复 |
| 多次调用 | 每次调用重新执行 | 从上次暂停处继续 |
| 适用场景 | 需要完整结果 | 需要流式/逐步产出 |

**最直观的理解**：把函数想象成做菜。
- `return` = 把所有菜都做好，**全部端上桌**（你饿着肚子等）
- `yield` = 做好一道菜就**先端上来**，你边吃边等下一道（不用饿肚子）

---

### 7.4 THINK 事件命名的误导性

**问题**：流式思考过程中，什么才算是 think？我问"上海天气怎么样"，LLM 返回"用户想要查询上海天气，我要调用 tool"，这不算是思考吗？但它放在了 content 里。而 `show_thinking=True` 跑"1+1等于几"反而有思考？

**解答**：这个问题暴露了当前代码中 **THINK 事件命名上的设计缺陷**。

看 `run_stream()` 第 206-217 行的实际代码：

```python
for chunk in self.llm.stream_chat(self.history, tools if tools else None):
    accumulator.add_chunk(chunk)
    new_text = accumulator.new_tokens()
    if new_text:
        yield StreamEvent(
            event=StreamEventType.THINK,  # ← 所有流式内容都叫 THINK
            data=new_text,
            step=step,
        )
```

**实际上，这里产出的 THINK 事件就是 LLM 返回的所有流式内容**，不管它是"思考过程"还是"最终回复"。因为在流式结束前，代码**无法区分**这次 LLM 调用最终是工具调用还是直接回复。

所以对于"上海天气怎么样"这个查询，LLM 流式返回的内容（全部被标记为 THINK）可能是：
```
"用户想要查询上海天气，我需要调用 get_weather 工具..."
```

对于"1+1等于几"这个查询，LLM 流式返回的内容（也全部被标记为 THINK）可能是：
```
"1+1等于2"
```

**两者在代码层面没有任何区别**——都是 `yield StreamEvent(THINK, data=...)`。

那为什么"1+1"看起来有思考效果？关键在于 `StreamingCallbackHandler` 的 `show_thinking` 参数：

```python
def _handle_think(self, event) -> None:
    if self.mode == "verbose":
        print(f"[思考] {event.data}", end="", flush=True)
    elif self.mode == "realtime" and self.show_thinking:  # ← 关键！
        print(f"\033[90m{event.data}\033[0m", end="", flush=True)  # 灰色显示
```

- `show_thinking=True`：THINK 事件的内容会以**灰色**打印出来
- `show_thinking=False`（默认）：THINK 事件的内容**不显示**

所以两个演示的差异只是 `show_thinking` 参数不同，而不是"上海天气没有思考过程，1+1有思考过程"。

**更准确的名字应该是 `STREAMING` 或 `TOKEN`**，因为：

| 事件名 | 实际含义 | 应该叫啥 |
|-------|---------|---------|
| `THINK` | LLM 流式返回的**所有文本 token**（不管是不是思考） | `STREAMING` 或 `TOKEN` |
| `REPLY` | 流式结束后，**重新逐字输出**的最终回复 | 这个叫 REPLY 还算合理 |

**真正的"思考过程"是什么？**

在 LLM 的 Function Calling 场景中，当 LLM 决定调用工具时，它返回的 `content` 字段可能是：
- `null`（很多模型直接返回 null，工具调用信息在 `tool_calls` 字段）
- 或者一段解释文字，比如"用户想查询上海天气，我来调用 get_weather 工具"

这段文字**确实可以理解为"思考过程"**，但代码并没有做任何智能判断——它只是把所有流式内容都标记为 THINK 而已。

---

### 7.5 实时 Token 计数

**问题**：代码中有没有流式过程中实时计算 token 的功能？像 Claude Code 那样思考过程中 token 数字在跳动。

**解答**：当前代码有 Token 统计，但不是实时跳动的。

`TokenCountingCallback`（`agent/callback/token_counting.py`）统计的是**每次 LLM 调用完成后的总 Token 数**，而不是流式过程中的实时 Token 数：

```python
def on_llm_end(self, response) -> None:
    """LLM 调用完成时，从 response 中提取 usage 信息并累加。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    self._prompt_tokens += usage.prompt_tokens
    self._completion_tokens += usage.completion_tokens
```

它依赖 OpenAI 返回的 `usage` 字段，这个字段**只在流式结束后才出现**。

不过 `StreamingCallbackHandler` 已经有实时计数的**基础**了：

```python
@property
def token_count(self) -> int:
    """已接收的文本 token 总数。"""
    return self._token_count
```

它在 `_handle_think` 和 `_handle_reply` 中都会累加字符数，但**没有实时显示**，只是在最后打印统计结果。

**要实现类似 Claude Code 的实时跳动效果**，只需在 `StreamingCallbackHandler` 中加几行代码：

```python
def _handle_think(self, event) -> None:
    self._token_count += len(str(event.data or ""))
    if self.show_token_count:  # 新增参数
        # 实时显示 Token 计数（\r 覆盖上一行）
        print(f"\r[思考中... 已接收 {self._token_count} tokens]", end="", flush=True)
    # ... 原有显示逻辑 ...
```

这样在流式过程中，终端会实时显示 `[思考中... 已接收 42 tokens]`，数字随着每个 THINK 事件不断跳动增加。

---

*本文档对应代码版本：2026-05-18*
