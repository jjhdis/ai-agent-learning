# 第八阶段总结 —— 高级 Agent 类型

> 对应代码：`agent/plan_execute/`、`agent/self_ask/`、`agent/xml_agent/`、`agent/core/agent.py`
> 演示文件：`demo_plan_execute.py`、`demo_phase8b.py`

---

## 一、五种 Agent 的核心循环对比

所有 Agent 本质上都是 **"LLM 循环 + 工具调用"** 模式，区别在于**通信协议**不同：

```
Agent 核心模式 = while 循环:
    1. 调 LLM
    2. 检查输出中是否有工具调用
    3. 有 → 执行工具 → 结果追加到对话 → 继续循环
    4. 无 → 这就是最终回答 → 返回
```

### 1.1 ReAct Agent（你的基础 Agent）

**文件**: `agent/core/agent.py`

```
工具调用方式: OpenAI API 的 tools 参数 → LLM 返回 tool_calls JSON
解析方式:    API 自动解析（结构化 JSON）
结果回传:    {"role": "tool", "tool_call_id": tc.id}
模型要求:    必须支持 Function Calling（GPT、DeepSeek 等）
```

**核心代码**（第 111-165 行）:
```python
for step in range(self.max_iterations):
    response = self.llm.chat(self.history, tools)  # tools 通过 API 参数传递
    msg = response.choices[0].message
    if not msg.tool_calls:
        return msg.content  # 最终回复
    self.history.append(msg.model_dump())
    self._handle_tool_calls(msg.tool_calls)  # 执行工具
```

**特点**:
- ✅ 最可靠 —— API 保证格式正确
- ✅ 支持多参数、多工具
- ❌ 模型必须支持 Function Calling
- ❌ 依赖特定 API 格式

---

### 1.2 Conversational Agent（带记忆的对话 Agent）

**文件**: `agent/core/agent.py` + `agent/memory/`

```
本质: ReAct Agent + Memory = Conversational Agent
不需要新增任何 Agent 类型，给 ReAct Agent 加上 Memory 就是 Conversational Agent。
```

**核心代码**（`agent/core/agent.py` 第 92-170 行）:
```python
def run(self, user_message: str):
    if self.memory:
        self.history = self.memory.load()  # 从记忆恢复历史
    self.history.append({"role": "user", "content": user_message})
    
    # ... ReAct 循环 ...
    
    self.history.append({"role": "assistant", "content": msg.content})
    if self.memory:
        self.memory.save(self.history)  # 保存到记忆
    return parsed
```

**三种记忆模式**（`agent/memory/` 目录）:

| 记忆类型 | 文件 | 行为 |
|---------|------|------|
| `ConversationBufferMemory` | `buffer.py` | 完整保存所有对话历史 |
| `ConversationBufferWindowMemory` | `window.py` | 只保留最近 K 轮对话 |
| `ConversationSummaryMemory` | `summary.py` | 用 LLM 摘要压缩历史，省 Token |

**特点**:
- ✅ **跨轮上下文** —— 第二句"北京呢？"知道"呢"指的是"天气"
- ✅ **零新增代码** —— 给 ReAct Agent 加 `memory` 参数即可
- ✅ **三种记忆策略** —— 按场景选择（完整/窗口/摘要）
- ❌ **Token 累积** —— 对话越长，每次调 LLM 的 context 越大
- ❌ **摘要记忆有信息损失** —— LLM 摘要可能遗漏细节

**适用场景**: 聊天机器人、客服系统、任何需要多轮对话的场景

**演示代码**（`demo_phase8b.py` 第 235-266 行）:
```python
memory = ConversationBufferMemory()
agent = Agent(llm_client=llm, registry=registry, memory=memory)

# 第1轮
agent.run("上海今天天气怎么样？")  # 记住了

# 第2轮 —— 依赖上一轮上下文
result = agent.run("那北京呢？")   # 知道"呢"=天气
```

---

### 1.3 Plan & Execute Agent（先规划再执行）


**文件**: `agent/plan_execute/agent.py`

```
这不是一个"循环"，而是三个阶段:
    Phase 1 - Planner:    LLM 分析任务，生成有序步骤列表
    Phase 2 - Executor:   逐步执行，每步创建一个 ReAct Agent
    Phase 3 - Summarizer: LLM 汇总所有执行结果
```

**核心代码**（第 53-98 行）:
```python
def run(self, task: str) -> str:
    # Phase 1: 制定计划
    plan = self._generate_plan(task)  # 一次 LLM 调用
    
    # Phase 2: 逐步执行
    results = self._execute_plan(plan)  # 每步一个 ReAct Agent
    
    # Phase 3: 汇总结果
    final_answer = self._summarize(task, plan, results)  # 一次 LLM 调用
    return final_answer
```

**执行器细节**（第 138-183 行）:
```python
def _execute_plan(self, plan: list[str]) -> list[dict]:
    results = []
    context_parts: list[str] = []
    for i, step in enumerate(plan, 1):
        # 为每一步创建独立的 ReAct Agent（隔离历史）
        executor = Agent(llm_client=self.llm, registry=self.registry)
        result = executor.run(full_step)
        results.append({"step": step, "result": result_str})
        context_parts.append(f"步骤{i}: {step}\n结果: {result_str}")
    return results
```

**特点**:
- ✅ **全局可见性** —— Planner 能看到所有步骤，不会"走一步忘一步"
- ✅ **可解释性** —— 用户能看到完整计划，可提前纠正方向
- ✅ **失败隔离** —— 某一步失败不影响其他步骤
- ✅ **降级方案** —— `_fallback_execute()` 在规划失败时降级为 ReAct（第 216 行）
- ❌ **Token 消耗极大** —— 1（规划）+ N×M（每步 M 次 ReAct 循环）+ 1（汇总）
- ❌ **延迟高** —— 串行执行所有步骤
- ❌ **不适合简单任务** —— 杀鸡用牛刀

**适用场景**: 复杂多步任务（旅行规划、研究报告、数据分析流程）

---

### 1.4 Self-Ask Agent（纯文本追问模式）


**文件**: `agent/self_ask/agent.py`

```
工具调用方式: System Prompt 约定 "Follow up: <查询>" 格式
解析方式:    正则表达式 _extract_follow_up() / _extract_final_answer()
结果回传:    "Intermediate answer: <结果>" 文本追加到 context
模型要求:    任何 LLM 都能用
```

**核心代码**（第 49-116 行）:
```python
def run(self, question: str) -> str:
    context = f"Question: {question}\n"
    for i in range(self.max_follow_ups):
        messages.append({"role": "user", "content": context})
        response = self.llm.chat(messages)
        text = response.choices[0].message.content
        
        final_answer = _extract_final_answer(text)  # 正则匹配 "So the final answer is:"
        if final_answer:
            return final_answer
        
        follow_up = _extract_follow_up(text)  # 正则匹配 "Follow up:"
        if follow_up:
            search_result = self.search_fn(follow_up)
            context += f"{text}\nIntermediate answer: {search_result}\n"
```

**特点**:
- ✅ **模型无关** —— 任何 LLM 都能用（Llama、Mistral 等）
- ✅ **实现极简** —— 两个正则表达式 + 一个 search_fn 函数指针
- ❌ **只能搜索** —— 不支持多参数工具，只能传一个查询字符串
- ❌ **解析不可靠** —— 正则匹配不到就乱了
- ❌ **context 越来越长** —— 每次追问都追加完整上下文

**适用场景**: 需要多步推理的搜索型问题，模型不支持 Function Calling 时

---

### 1.5 XML Agent（XML 标签驱动工具调用）


**文件**: `agent/xml_agent/agent.py`

```
工具调用方式: System Prompt 约定 <tool_call> XML 标签格式
解析方式:    正则表达式 _extract_tool_calls() 解析 XML
结果回传:    <tool_result> XML 标签文本追加到对话
模型要求:    任何 LLM 都能用
```

**核心代码**（第 64-143 行）:
```python
def run(self, task: str) -> str:
    messages = [{"role": "system", "content": self._system_prompt}]
    messages.append({"role": "user", "content": task})
    
    for i in range(self.max_iterations):
        response = self.llm.chat(messages)
        text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": text})
        
        tool_calls = _extract_tool_calls(text)  # 正则解析 XML
        if not tool_calls:
            return text  # 最终回复
        
        for tc in tool_calls:
            result = self.tool_executor(tc["name"], json.loads(tc["arguments"]))
            messages.append({"role": "user", "content": _format_tool_result(name, result)})
```

**System Prompt 中的 XML 格式约定**（第 205-232 行）:
```
当你需要使用工具时，请严格使用以下 XML 格式：
<tool_call>
<name>工具名称</name>
<arguments>JSON 格式的参数</arguments>
</tool_call>
```

**特点**:
- ✅ **模型无关** —— 任何 LLM 都能用
- ✅ **支持多参数** —— arguments 里传 JSON 字符串
- ✅ **支持多工具并行** —— 一次可以输出多个 `<tool_call>` 标签
- ✅ **可读性强** —— XML 比 JSON Schema 更直观
- ❌ **解析不可靠** —— 正则解析 XML 可能失败
- ❌ **prompt 占用** —— 工具描述和 XML 格式说明占用大量 context

**适用场景**: 模型不支持 Function Calling，但需要调用多参数工具

---

## 二、五种 Agent 的全面对比

| 维度 | ReAct Agent | Conversational | Plan & Execute | Self-Ask Agent | XML Agent |
|------|:-----------:|:--------------:|:--------------:|:--------------:|:---------:|
| **核心模式** | 单循环 | ReAct + Memory | 三阶段 | 单循环 | 单循环 |
| **工具调用格式** | API JSON (`tool_calls`) | API JSON | 内部用 ReAct | 纯文本 (`Follow up:`) | XML 标签 (`<tool_call>`) |
| **模型要求** | 需支持 FC | 需支持 FC | 需支持 FC | 任何模型 | 任何模型 |
| **多参数支持** | ✅ 原生 | ✅ 原生 | ✅ 内部 ReAct 支持 | ❌ 只能传字符串 | ✅ arguments 传 JSON |
| **多工具并行** | ✅ 原生 | ✅ 原生 | ✅ 内部 ReAct 支持 | ❌ 一次只能追问一个 | ✅ 一次多个 `<tool_call>` |
| **解析可靠性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **实现复杂度** | 中 | 中（+Memory） | 高 | 低 | 中 |
| **Token 消耗** | 中 | 中（随轮次增长） | **极高** | 中 | 中 |
| **延迟** | 中 | 中 | **高** | 中 | 中 |
| **跨轮记忆** | ❌ 每次独立 | ✅ 三种记忆策略 | ❌ 每次独立 | ❌ 每次独立 | ❌ 每次独立 |
| **全局规划能力** | ❌ 走一步看一步 | ❌ 走一步看一步 | ✅ 先规划再执行 | ❌ 逐层追问 | ❌ 走一步看一步 |
| **失败隔离** | ❌ 一步失败全崩 | ❌ 一步失败全崩 | ✅ 步骤独立 | ❌ 追问失败全崩 | ❌ 一步失败全崩 |
| **可解释性** | 中 | 中 | **高**（计划可见） | 中 | 中 |
| **降级方案** | ❌ 无 | ❌ 无 | ✅ `_fallback_execute()` | ❌ 无 | ❌ 无 |


---

## 三、Token 消耗对比（以旅行规划为例）

假设一个任务需要：查天气(1次) → 查攻略(1次) → 综合分析(1次)

### ReAct Agent
```
第1轮: LLM 思考 → 调天气工具 → 拿到结果
第2轮: LLM 思考 → 调攻略工具 → 拿到结果
第3轮: LLM 思考 → 生成最终回答
总计: 3 次 LLM 调用
```

### Plan & Execute Agent
```
Phase 1 - Planner:  1 次 LLM 调用（制定计划）
Phase 2 - Executor:
  步骤1(查天气): ReAct Agent 内部 2 次 LLM 调用
  步骤2(查攻略): ReAct Agent 内部 2 次 LLM 调用
  步骤3(综合分析): ReAct Agent 内部 1 次 LLM 调用
Phase 3 - Summarizer: 1 次 LLM 调用（汇总结果）
总计: 1 + (2+2+1) + 1 = 7 次 LLM 调用
```

### Self-Ask Agent
```
第1轮: LLM 输出 "Follow up: 上海天气" → 搜索 → 反馈结果
第2轮: LLM 输出 "Follow up: 上海攻略" → 搜索 → 反馈结果
第3轮: LLM 输出 "So the final answer is: ..."
总计: 3 次 LLM 调用
```

### XML Agent
```
第1轮: LLM 输出 <tool_call>天气</tool_call> → 执行 → 反馈结果
第2轮: LLM 输出 <tool_call>攻略</tool_call> → 执行 → 反馈结果
第3轮: LLM 输出最终回答（无 <tool_call>）
总计: 3 次 LLM 调用
```

---

## 四、本质总结

### 4.1 所有 Agent 都是"提示词工程 + 解析方式"的组合

```
Agent = System Prompt（约定格式） + LLM 循环 + 输出解析
```

| Agent 类型 | System Prompt 约定 | 解析方式 |
|-----------|-------------------|---------|
| ReAct | OpenAI tools 参数（API 级别） | API 自动解析 |
| Conversational | OpenAI tools 参数 + 记忆管理 | API 自动解析 + Memory |
| Self-Ask | "Follow up:" / "So the final answer is:" | 正则表达式 |
| XML | `<tool_call>` / `<tool_result>` 标签 | 正则表达式 |
| Plan & Execute | 规划提示词 + 汇总提示词 | 内部 ReAct Agent |

### 4.2 Conversational Agent 是"ReAct + Memory"

Conversational Agent 不需要新增任何 Agent 类型，它只是给 ReAct Agent 加了一个 `memory` 参数：

```
Conversational Agent = ReAct Agent + Memory
```

三种记忆策略对应不同场景：
- **Buffer**：对话轮次少，需要完整上下文
- **Window**：对话轮次多，只关心最近几轮
- **Summary**：对话极长，用摘要压缩省 Token

### 4.3 Plan & Execute 是"元模式"

P&E 不是另一种"通信协议"，而是**在 ReAct 之上加了一层编排**：

```
Plan & Execute = Planner(LLM) + Executor(多个 ReAct Agent) + Summarizer(LLM)
```

它不改变"LLM 怎么调工具"，而是改变"什么时候调、调完怎么汇总"。

### 4.4 选择建议

```
需要多轮对话?
├─ 是 → Conversational Agent（ReAct + Memory）
└─ 否 → 模型支持 Function Calling?
        ├─ 是 → 任务复杂需要全局规划?
        │       ├─ 是 → Plan & Execute（多花 token 换可靠性）
        │       └─ 否 → ReAct Agent（最可靠、最省 token）
        └─ 否 → 需要多参数工具?
                ├─ 是 → XML Agent（支持多参数）
                └─ 否 → Self-Ask Agent（最简单）
```

### 4.5 一句话总结

> **ReAct 是基础，Conversational 是 ReAct + Memory，P&E 是 ReAct 的编排，Self-Ask 和 XML 是 ReAct 在不同模型上的"降级实现"。**
> 它们底层都是同一个"LLM 循环 + 工具调用"模式，区别只在于"LLM 和 Agent 之间用什么语言沟通"。

