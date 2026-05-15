# 第三阶段总结：回调系统（Callback）

> **学习目标**：理解事件驱动架构，掌握回调系统的设计思想，学会通过回调扩展 Agent 功能而不修改核心代码。

---

## 目录

1. [什么是回调系统](#1-什么是回调系统)
2. [代码架构](#2-代码架构)
3. [核心机制：反射分发](#3-核心机制反射分发)
4. [已实现的回调处理器](#4-已实现的回调处理器)
5. [如何自定义回调](#5-如何自定义回调)
6. [多个回调协同工作](#6-多个回调协同工作)
7. [回调系统的设计思想](#7-回调系统的设计思想)
8. [与 LangChain 的对比](#8-与-langchain-的对比)

---

## 1. 什么是回调系统

### 1.1 直观理解

回调系统 = **事件通知机制**，就像在 Agent 的各个关键节点安装了"传感器"：

```
Agent 运行过程：
  收到消息 ──→ 调 LLM ──→ LLM 返回 ──→ 调工具 ──→ 工具返回 ──→ 再调 LLM ──→ 最终回复
                │            │            │            │
                ▼            ▼            ▼            ▼
            传感器1       传感器2       传感器3       传感器4
           (on_llm_start)(on_llm_end)(on_tool_start)(on_tool_end)
```

每个传感器（回调）只负责**记录/通知**，不控制流程。

### 1.2 回调系统 ≠ "返回到某个节点"

| 概念 | 比喻 | 说明 |
|---|---|---|
| **回调系统** | 监控摄像头 📹 | 只记录发生了什么，不干预流程 |
| **记忆系统** | 笔记本 📓 | 记录对话历史，可以翻回去看 |
| **返回到某个节点** | 时光机 ⏪ | 需要主动把状态回退到之前某个时刻 |

**回调系统不做流程控制**，它只是通知你"发生了什么"。

### 1.3 回调系统的本质

回调系统的本质是：**在不修改 Agent 核心代码的前提下，让外部代码在特定时机执行特定操作。**

这就是**控制反转（IoC）**——不是 Agent 主动去调用外部功能，而是 Agent 说"我到了某个节点了"，让外部代码自己决定要做什么。

---

## 2. 代码架构

回调系统由 3 个核心文件组成：

```
agent/callback/
├── base.py              # 基类：定义所有可监听的事件
├── manager.py           # 管理器：管理多个回调，分发事件
├── logging.py           # 具体实现：日志与追踪
├── token_counting.py    # 具体实现：Token 消耗统计
└── __init__.py          # 导出
```

### 2.1 事件分类

| 分类 | 事件 | 触发时机 |
|---|---|---|
| **Agent 级** | `on_agent_start` | Agent 开始处理用户消息 |
| | `on_agent_end` | Agent 返回最终回复 |
| | `on_agent_error` | Agent 执行出错 |
| **LLM 级** | `on_llm_start` | 即将调用 LLM |
| | `on_llm_end` | LLM 调用完成 |
| | `on_llm_error` | LLM 调用出错 |
| | `on_think` | LLM 输出推理内容（非最终回复） |
| **Tool 级** | `on_tool_start` | 即将执行工具 |
| | `on_tool_end` | 工具执行完成 |
| | `on_tool_error` | 工具执行出错 |

### 2.2 事件触发流程

```
Agent.run()
  │
  ├── self.callbacks.on_agent_start(message)
  │     └── _dispatch("on_agent_start", message)
  │           ├── handler1.on_agent_start(message)   ← LoggingCallback 记录日志
  │           ├── handler2.on_agent_start(message)   ← TokenCountingCallback 重置计数器
  │           └── ...
  │
  ├── self.callbacks.on_llm_start(history, tools)
  │     └── _dispatch("on_llm_start", history, tools)
  │           ├── handler1.on_llm_start(...)          ← 记录 LLM 调用开始
  │           └── ...
  │
  ├── response = self.llm.chat(...)
  │
  ├── self.callbacks.on_llm_end(response)
  │     └── _dispatch("on_llm_end", response)
  │           ├── handler1.on_llm_end(...)            ← 记录 LLM 返回
  │           ├── handler2.on_llm_end(...)            ← 提取 usage 统计 Token
  │           └── ...
  │
  ├── self.callbacks.on_think(content)               ← 如果有推理内容
  │
  ├── self.callbacks.on_tool_start(name, args)       ← 如果需要调工具
  │
  ├── self.callbacks.on_tool_end(name, result)
  │
  └── self.callbacks.on_agent_end(reply)
        └── _dispatch("on_agent_end", reply)
              ├── handler1.on_agent_end(...)          ← 打印耗时统计
              ├── handler2.on_agent_end(...)          ← 打印 Token 汇总
              └── ...
```

---

## 3. 核心机制：反射分发

### 3.1 `getattr` 反射

`CallbackManager._dispatch()` 是回调系统的核心，它使用 Python 的反射机制来调用方法：

```python
def _dispatch(self, event: str, *args, **kwargs):
    """向所有处理器分发事件，单个处理器的异常不中断其他处理器。"""
    for handler in self.handlers:
        try:
            method = getattr(handler, event, None)  # ← 关键：反射获取方法
            if method:
                method(*args, **kwargs)
        except Exception:
            pass  # 某个处理器出错，不影响其他处理器
```

### 3.2 `getattr(obj, name, default)` 详解

```python
method = getattr(handler, event, None)
```

这行代码的意思是：

> **从 `handler` 对象上获取名为 `event` 的属性，如果不存在就返回 `None`**

**示例**：

```python
# 当 _dispatch("on_llm_end", response) 被调用时
event = "on_llm_end"

# 对 LoggingCallback 实例：
method = getattr(logging_handler, "on_llm_end", None)
# → method = logging_handler.on_llm_end  （存在，是一个方法）
# → method(response)  → 调用 LoggingCallback.on_llm_end(response)

# 对 TokenCountingCallback 实例：
method = getattr(token_handler, "on_llm_end", None)
# → method = token_handler.on_llm_end  （也存在）
# → method(response)  → 调用 TokenCountingCallback.on_llm_end(response)
```

### 3.3 为什么用 `getattr` 而不是直接调用？

| 方式 | 问题 |
|---|---|
| `handler.on_llm_end(response)` | 如果 handler 没有这个方法，直接报 `AttributeError` |
| `getattr(handler, "on_llm_end", None)` | 安全，没有就返回 None，不会报错 |

### 3.4 异常隔离

```python
try:
    method = getattr(handler, event, None)
    if method:
        method(*args, **kwargs)
except Exception:
    pass  # 某个处理器出异常，不影响其他处理器
```

这意味着：**即使某个回调处理器崩溃了，其他处理器和 Agent 本身不受影响。**

---

## 4. 已实现的回调处理器

### 4.1 LoggingCallback（日志与追踪）

**文件**：`agent/callback/logging.py`

**功能**：记录 Agent 运行日志和性能统计。

| 统计指标 | 说明 |
|---|---|
| `elapsed` | 总耗时（秒） |
| `llm_call_count` | LLM 调用次数 |
| `tool_call_count` | 工具调用次数 |

**输出示例**：
```
[Callback] Agent 启动 — 用户消息: 上海今天天气怎么样？
[Callback] LLM #1 开始 — 消息数: 1, 可用工具: ['get_weather']
[Callback] LLM #1 返回 — content 长度: 0, tool_calls 数: 1
[Callback] 工具 #1 开始 — get_weather({'city': '上海'})
[Callback] 工具 #1 完成 — get_weather: 上海今天28°C...
[Callback] LLM #2 开始 — 消息数: 4, 可用工具: ['get_weather']
[Callback] LLM #2 返回 — content 长度: 50, tool_calls 数: 0
[Callback] Agent 完成 — 耗时 3.21s, LLM 调用 2 次, 工具调用 1 次
```

### 4.2 TokenCountingCallback（Token 消耗统计）

**文件**：`agent/callback/token_counting.py`

**功能**：统计每次 Agent 运行的 Token 消耗，自动计算预估费用。

| 统计指标 | 说明 |
|---|---|
| `prompt_tokens` | 输入 Token 总数 |
| `completion_tokens` | 输出 Token 总数 |
| `total_tokens` | 总 Token 数 |
| `llm_call_count` | LLM 调用次数 |
| `estimated_cost` | 预估费用（元） |

**内置价格表**（单位：元/百万 Token）：

| 模型 | 输入价格 | 输出价格 |
|---|---|---|
| `deepseek-chat` | ¥1.0 | ¥2.0 |
| `deepseek-reasoner` | ¥4.0 | ¥16.0 |
| `gpt-4o` | ¥15.0 | ¥60.0 |
| `gpt-4o-mini` | ¥1.5 | ¥6.0 |
| `qwen-plus` | ¥2.0 | ¥6.0 |
| `qwen-max` | ¥20.0 | ¥60.0 |

**输出示例**：
```
[Token] LLM #1 — 模型: deepseek-v4-flash, 输入: 1,234 tokens, 输出: 56 tokens, 本次合计: 1,290 tokens
[Token] LLM #2 — 模型: deepseek-v4-flash, 输入: 1,456 tokens, 输出: 89 tokens, 本次合计: 1,545 tokens
[Token] ════════════════════════════════════════════
[Token]  Token 消耗统计
[Token]  ════════════════════════════════════════════
[Token]  模型: deepseek-v4-flash
[Token]  LLM 调用次数: 2
[Token]  输入 Token:        1,234
[Token]  输出 Token:           56
[Token]  总 Token:          1,290
[Token]  预估费用:     ¥0.001234
[Token]  ════════════════════════════════════════════
```

---

## 5. 如何自定义回调

### 5.1 三步法

```python
# 步骤1：继承 BaseCallbackHandler
class MyCallback(BaseCallbackHandler):
    
    # 步骤2：覆写你关心的事件方法
    def on_llm_start(self, messages, tools=None):
        print(f"要调 LLM 了，消息数: {len(messages)}")
    
    def on_tool_start(self, name, args):
        print(f"要调工具了: {name}")
    
    # 不关心的事件不用覆写，基类默认空操作

# 步骤3：注册到 CallbackManager
callbacks = CallbackManager()
callbacks.add_handler(MyCallback())
agent = Agent(..., callbacks=callbacks)
```

### 5.2 完整示例：人工审核回调

```python
class HumanReviewCallback(BaseCallbackHandler):
    """在敏感工具执行前要求人工确认"""
    
    def on_tool_start(self, name, args):
        if name == "execute_code":  # 执行代码前需要人工确认
            answer = input(f"确认执行 {args}? (y/n)")
            if answer != "y":
                raise PermissionError("用户取消了操作")
```

### 5.3 完整示例：缓存回调

```python
class CacheCallback(BaseCallbackHandler):
    """缓存 LLM 回复，相同问题不重复调用"""
    
    def __init__(self):
        self.cache = {}
    
    def on_llm_start(self, messages, tools=None):
        cache_key = str(messages) + str(tools)
        if cache_key in self.cache:
            # 直接返回缓存结果（需要配合 Agent 改造）
            print("命中缓存！")
```

---

## 6. 多个回调协同工作

### 6.1 同时注册多个回调

```python
callbacks = CallbackManager()
callbacks.add_handler(LoggingCallback(verbose=True))        # 记录日志
callbacks.add_handler(TokenCountingCallback(verbose=True))  # 统计 Token
callbacks.add_handler(MyCustomCallback())                   # 你的自定义回调

agent = build_agent(callbacks=callbacks)
```

### 6.2 事件分发顺序

当 Agent 触发一个事件时，`CallbackManager` 会**按注册顺序**依次调用所有 handler 的对应方法：

```
Agent 触发 on_llm_end(response)
  │
  ├── 1. LoggingCallback.on_llm_end(response)      ← 打印 LLM 返回信息
  ├── 2. TokenCountingCallback.on_llm_end(response) ← 提取 usage 统计 Token
  └── 3. MyCustomCallback.on_llm_end(response)      ← 你的自定义逻辑
```

### 6.3 各司其职

每个 Callback 只关心自己需要的事件，互不干扰：

| 回调 | 监听的事件 | 做的事情 |
|---|---|---|
| `LoggingCallback` | `on_agent_start`, `on_llm_start`, `on_llm_end`, `on_think`, `on_tool_start`, `on_tool_end`, `on_agent_end` | 打印日志、统计耗时和调用次数 |
| `TokenCountingCallback` | `on_agent_start`, `on_llm_end`, `on_agent_end` | 统计 Token 消耗和费用 |
| `StreamingCallback`（假设） | `on_think` | 逐字推送内容到前端 |
| `ReviewCallback`（假设） | `on_tool_start` | 敏感操作前要求人工确认 |

---

## 7. 回调系统的设计思想

### 7.1 开闭原则

```
┌─────────────────────────────────────────┐
│           Agent 核心代码                 │
│  （不需要修改，保持稳定）                  │
│                                          │
│  收到消息 → 调 LLM → 调工具 → 返回结果    │
│              │         │                 │
│              ▼         ▼                 │
│         on_llm_start  on_tool_start      │
└──────────────┼─────────────┼─────────────┘
               │             │
     ┌─────────▼─────────────▼─────────┐
     │       回调系统（可扩展）           │
     │                                  │
     │  LoggingCallback  ← 记录日志     │
     │  StreamingCallback ← 流式推送    │
     │  ReviewCallback   ← 人工审核     │
     │  MetricsCallback  ← 统计监控     │
     │  CacheCallback    ← 缓存加速     │
     └──────────────────────────────────┘
```

**核心价值**：你想加任何功能，**不需要改 Agent 的代码**，只需要写一个新的 Callback 类注册进去就行。

### 7.2 回调系统能做什么

| 你要做的事情 | 在哪个回调里做 |
|---|---|
| 记录日志 | `on_llm_start` / `on_agent_end` |
| 流式输出 | `on_think` |
| 人工审核 | `on_tool_start` |
| 统计 Token | `on_llm_end` |
| 缓存命中 | `on_llm_start`（拦截并返回缓存） |
| 异常告警 | `on_agent_error` |
| 限流控制 | `on_llm_start`（检查是否超限） |
| 费用统计 | `on_agent_end`（汇总所有 Token） |

**回调系统 = 给 Agent 装了一排"插座"，你想插什么功能就插什么。**

---

## 8. 与 LangChain 的对比

| 概念 | 你的代码 | LangChain |
|---|---|---|
| 事件基类 | `BaseCallbackHandler`（11 个事件） | `BaseCallbackHandler`（20+ 个事件） |
| 事件管理器 | `CallbackManager._dispatch()` | `CallbackManager.on_llm_start()` 等 |
| 日志回调 | `LoggingCallback` | `ConsoleCallbackHandler` |
| Token 统计 | `TokenCountingCallback` | 无内置，需自定义 |
| 流式回调 | 未实现 | `StreamingCallbackHandler` |
| 追踪平台 | 无 | LangSmith |

你的代码已经实现了 LangChain 回调系统的**核心设计思想**，只是事件数量较少。后续可以根据需要扩展更多事件类型。

---

## 附录：关键代码速查

### BaseCallbackHandler（基类）

```python
class BaseCallbackHandler:
    # Agent 级
    def on_agent_start(self, message: str): ...
    def on_agent_end(self, reply: str): ...
    def on_agent_error(self, error: Exception): ...
    
    # LLM 级
    def on_llm_start(self, messages: list[dict], tools: list[dict] = None): ...
    def on_llm_end(self, response): ...
    def on_llm_error(self, error: Exception): ...
    def on_think(self, content: str): ...
    
    # Tool 级
    def on_tool_start(self, name: str, args: dict): ...
    def on_tool_end(self, name: str, result: str): ...
    def on_tool_error(self, name: str, error: Exception): ...
```

### CallbackManager（管理器）

```python
class CallbackManager:
    def __init__(self, handlers=None):
        self.handlers = list(handlers) if handlers else []
    
    def add_handler(self, handler): ...
    def remove_handler(self, handler): ...
    
    def _dispatch(self, event, *args, **kwargs):
        for handler in self.handlers:
            try:
                method = getattr(handler, event, None)
                if method:
                    method(*args, **kwargs)
            except Exception:
                pass  # 异常隔离
```

### 在 Agent 中使用

```python
class Agent:
    def __init__(self, ..., callbacks=None):
        self.callbacks = callbacks or CallbackManager()
    
    def run(self, user_message):
        self.callbacks.on_agent_start(user_message)
        # ... 业务逻辑 ...
        self.callbacks.on_llm_start(self.history, tools)
        response = self.llm.chat(...)
        self.callbacks.on_llm_end(response)
        # ... 更多逻辑 ...
        self.callbacks.on_agent_end(reply)
```

---

*本文档对应代码版本：2026-05-15*
