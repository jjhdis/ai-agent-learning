# 第七阶段总结 —— RAG 检索增强生成与工具机制

> 本文档详细记录了 RAG 的完整流程、每个环节的代码做了什么、以及 Agent 的工具调用机制。适合一周后回看依然能看懂。

---

## 一、RAG 是做什么的？

**RAG（Retrieval-Augmented Generation，检索增强生成）** 的核心思想是：

> **给 AI 一个"外挂知识库"，让它能查到训练数据里没有的信息。**

比如你有一个公司内部文档（员工手册、产品说明书），这些内容大模型在训练时没看过，所以直接问它它不知道。RAG 的做法是：

1. 先把这些文档**存进一个向量数据库**
2. 用户提问时，**从库里找出最相关的几段文本**
3. 把找到的文本**连同问题一起发给 AI**，让 AI 基于这些材料回答

这样 AI 就能回答原本不知道的问题了。

---

## 二、完整数据流全景（从文档到检索）

下面这个图展示了从原始文档到最终检索的完整流程，每个步骤我都标注了对应的代码文件和核心方法：

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG 完整数据流                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  原始文档（.txt / .md / 字符串列表）                             │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────────────────────┐                       │
│  │ 第1步：创建 Document 对象             │                       │
│  │ 文件: agent/rag/document.py          │                       │
│  │ 类:   Document (数据类)              │                       │
│  │ 代码:                                │                       │
│  │   doc = Document(                    │                       │
│  │       content="很长一段文本...",      │                       │
│  │       metadata={"source": "xxx.txt"} │                       │
│  │   )                                  │                       │
│  │                                      │                       │
│  │ Document 只有两个字段：               │                       │
│  │  • content: 文本内容（字符串）         │                       │
│  │  • metadata: 元数据（字典），记录来源   │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────────────────────┐                       │
│  │ 第2步：分割文档（Chunking）            │                       │
│  │ 文件: agent/rag/splitter.py          │                       │
│  │ 类:   RecursiveCharacterTextSplitter │                       │
│  │ 调用:                                │                       │
│  │   chunks = splitter.split_documents( │                       │
│  │       docs                           │                       │
│  │   )                                  │                       │
│  │                                      │                       │
│  │ 为什么要分割？                        │                       │
│  │ 因为一篇文档可能很长（几千字），         │                       │
│  │ 如果整篇转成一个向量，检索精度会很差。    │                       │
│  │ 切成小块后，每块单独向量化，             │                       │
│  │ 检索时能精确找到相关段落。              │                       │
│  │                                      │                       │
│  │ 分割策略：递归字符分割                  │                       │
│  │ 按优先级尝试分隔符：                    │                       │
│  │ 段落(\n\n) → 换行(\n) → 句号(。)      │                       │
│  │ → 英文句点(.) → 双空格 → 单空格       │                       │
│  │                                      │                       │
│  │ 默认参数：                            │                       │
│  │  chunk_size = 500   （每块最多500字）   │                       │
│  │  overlap_size = 50  （相邻块重叠50字）  │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────────────────────┐                       │
│  │ 第3步：向量化并存储                    │                       │
│  │ 文件: agent/rag/store.py             │                       │
│  │ 类:   InMemoryVectorStore            │                       │
│  │ 调用:                                │                       │
│  │   store.add_documents(chunks)        │                       │
│  │                                      │                       │
│  │ add_documents 内部做了三件事：         │                       │
│  │  ① texts = [doc.content for doc in   │                       │
│  │           chunks]   ← 提取所有文本     │                       │
│  │                                      │                       │
│  │  ② vectors = embedding.embed_documents│                       │
│  │           (texts)  ← 文本→向量        │                       │
│  │                                      │                       │
│  │  ③ self._documents.extend(chunks)    │                       │
│  │     self._vectors.extend(vectors)     │                       │
│  │     ← 文档和向量分开存，通过索引对应    │                       │
│  │                                      │                       │
│  │ 存储结构：                            │                       │
│  │  _documents[0] ←→ _vectors[0]        │                       │
│  │  _documents[1] ←→ _vectors[1]        │                       │
│  │  ...                                 │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────────────────────┐                       │
│  │ 第4步：检索（用户提问时触发）          │                       │
│  │ 文件: agent/rag/retriever.py         │                       │
│  │ 类:   RetrieverTool                  │                       │
│  │ 调用:                                │                       │
│  │   result = tool.execute(query)       │                       │
│  │                                      │                       │
│  │ execute 内部：                        │                       │
│  │  ① 调 store.similarity_search(       │                       │
│  │       query, k=4)                    │                       │
│  │     ← 查向量库，返回最像的4个文档       │                       │
│  │                                      │                       │
│  │  ② 把结果格式化成可读文本              │                       │
│  │     ← 返回给 Agent，Agent 再给 LLM    │                       │
│  └──────────────────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、分割器（splitter）的详细工作原理

### 3.1 为什么需要分割？

假设你有一篇 5000 字的公司制度文档，如果不分割：
- 整篇转成一个向量 → 信息太杂，检索时找不到具体段落
- 超出模型的 token 限制 → 没法一次性处理

所以需要切成小块，每块 500 字左右，每块单独向量化。

### 3.2 分割策略：递归字符分割

代码在 `agent/rag/splitter.py` 的 `_split_text` 方法中：

```python
def _split_text(self, text: str, separators: list[str]) -> list[str]:
    # 1. 如果文本已经够短（≤ chunk_size），直接返回，不再切分
    stripped = text.strip()
    if len(stripped) <= self.chunk_size:
        return [stripped] if stripped else []

    # 2. 取出当前优先级的分隔符
    sep = separators[0] if separators else ""
    remaining = separators[1:] if len(separators) > 1 else []

    # 3. 如果当前分隔符不可用（不在文本中），降级到下一级
    if not sep or sep not in text:
        if remaining:
            return self._split_text(text, remaining)
        return self._force_split(text)  # 所有分隔符都用完了，强制按长度切

    # 4. 按分隔符切分，然后合并小片段
    parts = text.split(sep)
    result = []
    buffer = ""
    for part in parts:
        merged = buffer + (sep if buffer else "") + part
        if len(merged) <= self.chunk_size:
            buffer = merged  # 合并后没超长，继续合并
        else:
            if buffer.strip():
                result.append(buffer)  # 把 buffer 作为一块保存
            if len(part) > self.chunk_size:
                # 当前片段本身超长，递归细分
                result.extend(self._split_text(part, remaining))
                buffer = ""
            else:
                buffer = part  # 用当前片段作为新 buffer 起点
    if buffer.strip():
        result.append(buffer)
    return result
```

### 3.3 举个例子

假设 `chunk_size=10`，文本是 `"AAAA\n\nBBBB\n\nCCCCCCCCCCCC"`：

```
第1轮：用 "\n\n" 切分
  parts = ["AAAA", "BBBB", "CCCCCCCCCCCC"]
  
  处理 "AAAA" → buffer = "AAAA"（没超10字）
  处理 "BBBB" → merged = "AAAA\n\nBBBB" = 9字 ≤ 10 → buffer = "AAAA\n\nBBBB"
  处理 "CCCCCCCCCCCC" → merged = "AAAA\n\nBBBB\n\nCCCCCCCCCCCC" = 20字 > 10
    → 先把 buffer("AAAA\n\nBBBB") 作为一块保存
    → "CCCCCCCCCCCC" 本身 12字 > 10，递归处理
      → 用下一级分隔符 "\n" 切分（假设没有换行）
      → 用 "。" 切分（假设也没有句号）
      → ... 直到所有分隔符用完
      → _force_split() 强制按步长切分

最终结果：
  ["AAAA\n\nBBBB", "CCCCCCCC", "CCCC"]  （最后两个是强制切分的）
```

### 3.4 强制切分（兜底方案）

```python
def _force_split(self, text: str) -> list[str]:
    """所有分隔符都无效时的终极手段：无视语义，按固定步长切分。"""
    step = max(self.chunk_size - self.overlap_size, 1)  # 步长 = 500-50 = 450
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + self.chunk_size].strip()  # 每次取500字
        if chunk:
            chunks.append(chunk)
    return chunks
```

比如一段 1200 字的纯文本（没有段落、没有标点）：
- 步长 = 500 - 50 = 450
- 第1块：字符 0~500
- 第2块：字符 450~950（和上一块重叠50字）
- 第3块：字符 900~1200（和上一块重叠50字）

**重叠的作用**：避免在句子中间切断导致上下文丢失，重叠部分能保留一些连贯性。

---

## 四、Embedding（向量化）的底层原理

### 4.1 什么是 Embedding？

**Embedding 就是把"文本"变成"一串数字（向量）"的过程。**

比如：
```
"Python 是一种编程语言"  →  [0.023, -0.145, 0.678, ..., 0.332]  （1024维向量）
"Java 也是一种语言"      →  [0.018, -0.132, 0.654, ..., 0.301]  （语义相近，向量也相近）
"今天天气真好"           →  [-0.201, 0.456, -0.123, ..., 0.087]  （语义不同，向量也远离）
```

**关键特性**：语义相近的文本，得到的向量在数学上也相近（余弦相似度高）。

### 4.2 完整调用链（从代码到远程服务器）

```
┌─────────────────────────────────────────────────────────────────────┐
│               vectors = self._embedding.embed_documents(texts)      │
│               （store.py 第48行）                                    │
│                                                                     │
│  这行代码的背后，是一连串的调用：                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第1层：store.py（调用入口）                                          │
│                                                                     │
│ 代码：                                                              │
│   texts = [doc.content for doc in documents]   ← 先把所有文本拿出来  │
│   vectors = self._embedding.embed_documents(texts)  ← 调用下一层    │
│   self._documents.extend(documents)   ← 存原文                      │
│   self._vectors.extend(vectors)       ← 存向量                      │
│                                                                     │
│ 做的事情：把文档的文本内容提取出来，传给 embedding 对象，               │
│           拿到向量后和原文一起存起来。                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第2层：embedding.py（封装层）                                        │
│                                                                     │
│ 代码：                                                              │
│   def embed_documents(self, texts: list[str]) -> list[list[float]]: │
│       if not texts:                                                 │
│           return []                                                 │
│       resp = self._client.embeddings.create(                        │
│           model=self.model,    ← 比如 "embedding-2"（智谱的模型）    │
│           input=texts          ← 传入文本列表                        │
│       )                                                             │
│       sorted_data = sorted(resp.data, key=lambda x: x.index)        │
│       return [d.embedding for d in sorted_data]                     │
│                                                                     │
│ 做的事情：                                                           │
│   ① 调用 OpenAI SDK 的 embeddings.create() 方法                     │
│   ② 传入 model 和 input 参数                                        │
│   ③ 返回结果按 index 排序（保证顺序和传入的 texts 一致）              │
│   ④ 提取每个结果的 .embedding 字段（就是 list[float]）               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第3层：OpenAI SDK（HTTP 请求封装）                                    │
│ 文件：.venv/Lib/site-packages/openai/resources/embeddings.py        │
│                                                                     │
│ 代码（简化）：                                                       │
│   def create(self, ..., input, model, ...):                         │
│       params = {                                                    │
│           "input": input,          ← ["文本1", "文本2"]              │
│           "model": model,          ← "embedding-2"                  │
│           "encoding_format": "base64",  ← 用 base64 编码传输        │
│       }                                                             │
│       return self._post("/embeddings", body=params, ...)            │
│                                                                     │
│ 做的事情：                                                           │
│   ① 把参数打包成字典                                                 │
│   ② 通过 httpx 库发送 HTTP POST 请求到远程服务器                     │
│   ③ 请求体是 JSON 格式，大概长这样：                                  │
│      {                                                              │
│        "input": ["Python是一种编程语言", "Java也是一种语言"],          │
│        "model": "embedding-2",                                      │
│        "encoding_format": "base64"                                  │
│      }                                                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第4层：远程 AI 服务器（真正的 Embedding 计算）                        │
│                                                                     │
│ 这一步不在你的代码里，而是在智谱/OpenAI 的服务器上。服务器收到请求后：   │
│                                                                     │
│ ① 分词（Tokenization）                                              │
│    把文本拆成 token（词元）：                                        │
│    "Python是一种编程语言" → ["Python", "是", "一种", "编程", "语言"]  │
│                                                                     │
│ ② 神经网络推理                                                      │
│    把 token 序列输入到一个大型神经网络模型                              │
│    （比如 text-embedding-3-small）                                   │
│    这个模型经过海量文本训练，学会了把语义映射到向量空间                  │
│    输出是一个固定维度的向量（比如 1024 维或 1536 维）                  │
│                                                                     │
│ ③ 返回结果                                                          │
│    服务器返回 JSON 响应：                                            │
│    {                                                                │
│      "data": [                                                      │
│        {"index": 0, "embedding": "base64编码的二进制数据..."},        │
│        {"index": 1, "embedding": "base64编码的二进制数据..."}         │
│      ],                                                             │
│      "model": "embedding-2"                                         │
│    }                                                                │
│    为了节省带宽，向量数据用 base64 编码传输                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第5层：返回路径（base64 解码）                                       │
│                                                                     │
│ OpenAI SDK 收到响应后，在 parser 函数中做解码：                      │
│                                                                     │
│   def parser(obj):                                                  │
│       for embedding in obj.data:                                    │
│           data = embedding.embedding  ← 这是 base64 字符串           │
│           # 解码三步走：                                             │
│           # ① base64 → 二进制字节                                   │
│           # ② 二进制 → float32 数字数组                              │
│           # ③ numpy 数组 → Python list                              │
│           embedding.embedding = np.frombuffer(                      │
│               base64.b64decode(data), dtype="float32"               │
│           ).tolist()                                                │
│       return obj                                                    │
│                                                                     │
│ 最终你拿到的 vectors 就是：                                          │
│   [                                                               │
│     [0.023, -0.145, 0.678, ..., 0.332],  ← "Python是一种编程语言"   │
│     [0.018, -0.132, 0.654, ..., 0.301],  ← "Java也是一种语言"       │
│   ]                                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 一句话总结

**`vectors = self._embedding.embed_documents(texts)` 的本质就是：把你的文本通过 HTTP 请求发给远程的 AI 服务器，服务器用神经网络模型把文本"翻译"成一串数字（向量），然后返回给你。你不需要关心神经网络内部怎么算的，只需要知道：语义相近的文本，得到的数字向量也相近。**

---

## 五、检索（similarity_search）的原理

### 5.1 检索流程

```python
def similarity_search(self, query: str, k: int = 4) -> list[Document]:
    # 1. 把用户的查询也转成向量
    query_vec = self._embedding.embed_query(query)

    # 2. 遍历所有已存储的文档向量，计算相似度
    scores = []
    for i, vec in enumerate(self._vectors):
        score = self._cosine_similarity(query_vec, vec)
        scores.append((score, i))  # (相似度, 索引)

    # 3. 按相似度从高到低排序
    scores.sort(key=lambda x: x[0], reverse=True)

    # 4. 返回最相似的 k 个文档
    top_k = min(k, len(scores))
    return [self._documents[i] for _, i in scores[:top_k]]
```

**用大白话说就是：**

> 你有一堆文档，每个文档都对应一个"数字指纹"（向量）。现在用户问了一个问题，我把问题也转成"数字指纹"，然后跟所有文档的指纹比一比，看哪个最像，把最像的那几个拿出来给 AI。

### 5.2 余弦相似度公式

```python
@staticmethod
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """两个等长向量的余弦相似度，值域 [-1, 1]，越大越相似。"""
    dot = sum(x * y for x, y in zip(a, b))          # 点积：对应位置相乘再求和
    norm_a = math.sqrt(sum(x * x for x in a))        # 向量a的长度
    norm_b = math.sqrt(sum(x * x for x in b))        # 向量b的长度
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)                    # 余弦值
```

**直观理解**：两个向量的方向越一致，余弦值越接近 1；方向相反，越接近 -1；垂直（不相关），接近 0。

---

## 六、工具机制（Tool System）详解

### 6.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent 工具调用机制                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────────┐    │
│  │  用户提问  │────→│   Agent 引擎  │────→│   LLM（大模型）   │    │
│  └──────────┘     │  agent.py    │     │                  │    │
│                   │              │     │ 返回工具名+参数    │    │
│                   │              │←────│                  │    │
│                   └──────┬───────┘     └──────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│                   ┌──────────────┐                              │
│                   │ ToolRegistry  │                             │
│                   │ (注册中心)    │                             │
│                   │              │                              │
│                   │ 根据 name    │                              │
│                   │ 查找工具对象  │                              │
│                   └──────┬───────┘                              │
│                          │                                      │
│          ┌───────────────┼───────────────┐                      │
│          ▼               ▼               ▼                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │RetrieverTool│  │WeatherTool │  │Calculator  │  ← 每个工具    │
│  │            │  │            │  │            │     各自实现    │
│  │execute()   │  │execute()   │  │execute()   │     execute()  │
│  │向量检索     │  │调天气API   │  │数学计算     │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 BaseTool：所有工具的基类

```python
# agent/tools/base.py
class BaseTool(ABC):
    """所有工具必须继承这个基类。"""
    
    name: str = ""          # 工具名称，LLM 通过这个名字引用
    description: str = ""   # 工具描述，LLM 根据描述决定是否调用
    parameters: list = []   # 参数定义，告诉 LLM 需要传什么参数

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具逻辑，返回字符串结果。"""
        ...
```

**关键点**：`execute` 是一个抽象方法，每个子类必须自己实现。虽然都叫 `execute`，但不同工具做的事情完全不同。

### 6.3 to_openai_function()：把工具描述翻译给 LLM 听

```python
def to_openai_function(self) -> dict:
    """转换为 OpenAI function calling 格式。"""
    properties = {}
    required = []

    for p in self.parameters:
        prop = {"type": p.type, "description": p.description}
        if p.enum:
            prop["enum"] = p.enum
        properties[p.name] = prop
        if p.required:
            required.append(p.name)

    return {
        "type": "function",
        "function": {
            "name": self.name,          # ← 工具名
            "description": self.description,  # ← 工具描述
            "parameters": {
                "type": "object",
                "properties": properties,  # ← 参数列表
                "required": required,
            },
        },
    }
```

这个方法生成的 JSON 会发给 LLM，告诉 LLM：
- 有哪些工具可以用
- 每个工具是干什么的（description）
- 调用每个工具需要传什么参数（parameters）

**注意**：这个 JSON 里没有 `execute` 字段。LLM 根本不知道 `execute` 这个方法的存在。

### 6.4 两个具体工具的对比

**RetrieverTool（知识检索）：**
```python
class RetrieverTool(BaseTool):
    name = "retrieve_knowledge"
    description = "从知识库中检索与查询相关的信息..."
    parameters = [
        ToolParameter(name="query", type="string",
                      description="搜索查询词或具体问题"),
    ]

    def execute(self, query: str) -> str:
        # 里面是向量检索逻辑
        docs = self._store.similarity_search(query, k=self._k)
        # 格式化结果返回
        return 格式化后的检索结果
```

**WeatherTool（天气查询）：**
```python
class WeatherTool(BaseTool):
    name = "get_weather"
    description = "查询指定城市的实时天气和未来天气预报..."
    parameters = [
        ToolParameter(name="city", type="string",
                      description="城市名称，中文或英文均可"),
    ]

    def execute(self, city: str) -> str:
        # 里面是调用天气 API 的逻辑
        resp = requests.get(f"https://wttr.in/{city}", ...)
        return 格式化后的天气数据
```

**虽然都叫 `execute`，但里面干的完全是两码事：**
- 一个查向量数据库
- 一个调 HTTP 接口

这就是**多态**：同样的方法名，不同的对象执行不同的逻辑。

### 6.5 ToolRegistry：注册中心

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}  # 本质就是一个字典

    def register(self, tool: BaseTool) -> None:
        """注册一个工具。以 name 为 key 存进字典。"""
        self._tools[tool.name] = tool

    def get(self, name: str):
        """根据 name 从字典中取出工具对象。"""
        return self._tools.get(name)

    def get_openai_definitions(self) -> list[dict]:
        """返回所有工具的 OpenAI function calling 格式定义。"""
        return [t.to_openai_function() for t in self._tools.values()]
```

**注册时：**
```python
registry = ToolRegistry()
registry.register(RetrieverTool())    # → _tools["retrieve_knowledge"] = RetrieverTool对象
registry.register(WeatherTool())      # → _tools["get_weather"] = WeatherTool对象
```

### 6.6 Agent 调用工具的完整流程（agent.py）

```python
# agent.py 第337-366行
def _handle_tool_calls(self, tool_calls) -> None:
    """执行工具调用并将结果追加到历史。"""
    for tc in tool_calls:
        # 第1步：根据 name 从注册中心找到对应的 Tool 对象
        tool = self.registry.get(tc.function.name)
        # 比如 tc.function.name = "retrieve_knowledge"
        # 那么 tool = RetrieverTool 对象

        if not tool:
            content = f"[错误] 未知工具: {tc.function.name}"
        else:
            # 第2步：解析参数（LLM 返回的是 JSON 字符串）
            args = json.loads(tc.function.arguments)
            # 比如 args = {"query": "Python GIL"}

            # 第3步：调用 tool 对象的 execute() 方法
            content = tool.execute(**args)
            # 相当于 RetrieverTool.execute(query="Python GIL")

        # 第4步：把工具执行结果追加到对话历史
        self.history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": content,
        })
```

**完整流程用大白话说：**

```
第1步：Agent 把所有工具的定义（名字、描述、参数格式）发给 LLM
  ↓
第2步：LLM 看完描述后，决定"这个问题需要调用 retrieve_knowledge 工具"
  ↓
第3步：LLM 返回一个 JSON：
  {
    "name": "retrieve_knowledge",
    "arguments": '{"query": "Python GIL是什么"}'
  }
  ↓
第4步：Agent 代码收到这个返回，做两件事：
  ① registry.get("retrieve_knowledge") → 找到 RetrieverTool 对象
  ② tool.execute(query="Python GIL是什么") → 执行向量检索
  ↓
第5步：execute() 内部执行向量检索，返回结果字符串
  ↓
第6步：Agent 把结果字符串追加到对话历史，再发给 LLM 继续推理
```

**关键点**：LLM 只负责"选工具+传参数"，真正去执行 `execute()` 的是 Agent 代码。`to_openai_function()` 返回的 JSON 里没有 `execute` 字段，也不需要。

### 6.7 为什么每个工具一个类（而不是一个大接口）

**不好的写法（把所有工具放一个类里）：**
```python
class BigTool:
    def execute(self, tool_name, **kwargs):
        if tool_name == "retrieve_knowledge":
            # 向量检索逻辑...
        elif tool_name == "get_weather":
            # 天气查询逻辑...
        elif tool_name == "calculator":
            # 计算逻辑...
        # 每加一个工具就要在这里加一个 elif
```

这种写法的问题：
1. **每加一个工具就要改这个文件**，改着改着就出 bug
2. **文件会越来越臃肿**，10 个工具还好，100 个工具就几千行了
3. **工具之间耦合**，改天气工具可能不小心影响到检索工具

**好的写法（每个工具一个类）：**
```python
class RetrieverTool(BaseTool):  # 单独一个文件
    def execute(self, query): ...

class WeatherTool(BaseTool):    # 单独一个文件
    def execute(self, city): ...

class CalculatorTool(BaseTool): # 单独一个文件
    def execute(self, expr): ...
```

优点：
1. **加工具 = 新建一个文件**，继承 BaseTool 就行，不用改任何现有代码
2. **每个工具独立**，互不干扰
3. **符合开闭原则**：对扩展开放（加新工具），对修改关闭（不改旧代码）

### 6.8 这个设计模式叫什么？

这叫**策略模式（Strategy Pattern）**：

```
BaseTool（抽象策略）
  ├── RetrieverTool（具体策略1）→ execute = 向量检索
  ├── WeatherTool（具体策略2）  → execute = 调天气API
  └── CalculatorTool（具体策略3）→ execute = 数学计算

Agent（上下文）
  └── 根据 LLM 返回的 name，选择对应的策略执行
```

**核心思想**：定义一系列算法（工具），把它们一个个封装起来，让它们可以互相替换。

---

## 七、RAG vs 其他数据获取方式

### 7.1 各场景对比

| 场景 | 数据形式 | 获取方式 | 示例 |
|------|---------|---------|------|
| 查数据库 | 结构化表格 | SQL 查询 | `QuerySQLTool` |
| 查实时天气 | API 返回的 JSON | HTTP 请求 | `WeatherTool` |
| 查公开信息 | 网页 | 联网搜索 | `WebSearchTool` |
| **查内部文档/知识库** | **非结构化长文本** | **RAG 检索** | **`RetrieverTool`** |

### 7.2 RAG 适合的场景

- **企业知识库**：员工手册、政策文档、培训材料
- **产品文档**：技术规范、用户手册、FAQ
- **代码库问答**：项目有几百个文件，用 RAG 检索相关代码
- **学术论文检索**：导入论文 PDF，检索相关段落
- **任何"文本形式、无法用 SQL 精确查询"的知识**

### 7.3 RAG 不适合的场景

- **实时数据**（天气、新闻）→ 用 API/联网搜索
- **结构化数据**（数据库）→ 用 SQL 查询
- **模型本身已经掌握的知识** → 直接回答

### 7.4 实际项目中的搭配使用

一个成熟的 AI Agent 通常会同时具备多种能力：

```
Agent 收到用户问题
  ├─ 如果是实时信息（天气、新闻）→ 调用联网搜索工具
  ├─ 如果是内部知识（公司文档、代码）→ 调用 RAG 检索工具
  ├─ 如果是数据库查询 → 调用 SQL 查询工具
  └─ 如果是通用知识 → 直接靠模型自身知识回答
```

**RAG 只是工具箱里的一个选项**，和 SQL 查询、API 调用、联网搜索各司其职。

---

## 八、生产环境怎么做 RAG？（不用自己造轮子）

### 8.1 你项目代码 vs 生产环境

你现在自己写的 `InMemoryVectorStore` 是为了学习原理，生产环境**没人会自己写向量数据库和检索逻辑**。

| 你的代码（学习用） | 生产环境（用现成的框架/库） |
|------------------|---------------------------|
| `Document` | `langchain_core.documents.Document` |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters.RecursiveCharacterTextSplitter` |
| `OpenAIEmbeddings` | `langchain_openai.OpenAIEmbeddings` |
| `InMemoryVectorStore`（存 list 里） | Pinecone / Milvus / ChromaDB / pgvector |
| `_cosine_similarity`（自己写） | 向量数据库内置，不用自己写 |
| `RetrieverTool`（自己封装） | `langchain.tools.RetrieverTool` 或简单封装 |

**你学完这套代码，再去看 LangChain 的文档，会发现每个概念你都认识，只是 API 名字不同。**

### 8.2 生产环境用 LangChain 的代码示例

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. 加载文档
loader = TextLoader("公司文档.txt")
docs = loader.load()

# 2. 切分（和你项目里的逻辑一样）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 向量化 + 存到 Pinecone（一行代码搞定）
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    index_name="company-docs"
)

# 4. 检索（也是一行代码）
results = vectorstore.similarity_search("年假政策是什么？", k=4)
```

### 8.3 主流向量数据库对比

| 产品 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| **Pinecone** | 云服务 | 全托管，零运维，免费额度够学习 | 小团队快速上线 |
| **Milvus / Zilliz** | 开源/云 | 性能强，功能全，支持千亿级向量 | 大规模生产 |
| **Weaviate** | 开源/云 | 自带 schema 和推理模块 | 中小规模 |
| **Qdrant** | 开源/云 | Rust 写的，性能好，API 简洁 | 各种规模 |
| **ChromaDB** | 开源嵌入式 | 轻量，本地运行，pip install 就能用 | 学习和原型开发 |
| **pgvector** | PostgreSQL 插件 | 不用额外维护数据库，直接用 SQL 查向量 | 已有 PostgreSQL 的项目 |
| **Redis Stack** | Redis 模块 | 内存快，支持向量搜索 | 需要低延迟的场景 |

### 8.4 生产 RAG 的完整架构

```
┌─────────────────────────────────────────────────────────────┐
│                    生产环境 RAG 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  你的公司文档（PDF / Word / Wiki / 数据库）                    │
│    │                                                        │
│    ▼                                                        │
│  ┌─────────────────────┐                                    │
│  │ 文档加载器            │  ← LangChain 的 DocumentLoader    │
│  │ (PDFLoader, etc.)   │     自动解析各种格式                 │
│  └─────────┬───────────┘                                    │
│            ▼                                                │
│  ┌─────────────────────┐                                    │
│  │ 文本分割器            │  ← RecursiveCharacterTextSplitter │
│  │ (chunk_size=500)    │     （和你项目里的一样）             │
│  └─────────┬───────────┘                                    │
│            ▼                                                │
│  ┌─────────────────────┐                                    │
│  │ Embedding API       │  ← OpenAI / 智谱 / 本地开源模型     │
│  │ (文本 → 向量)        │     （和你项目里的一样）             │
│  └─────────┬───────────┘                                    │
│            ▼                                                │
│  ┌─────────────────────┐                                    │
│  │ 向量数据库           │  ← Pinecone / Milvus / pgvector   │
│  │ (存向量 + 检索)      │     （不用自己写，调 SDK 就行）      │
│  │                     │                                     │
│  │ 向量数据库内部做了：   │                                   │
│  │  • 向量索引（ANN）    │  ← 近似最近邻搜索，比暴力遍历快     │
│  │  • 持久化到磁盘       │  ← 重启不丢失                     │
│  │  • 分布式存储         │  ← 支持海量数据                   │
│  └─────────────────────┘                                    │
│                                                             │
│  用户提问时的检索流程：                                       │
│  ① 问题 → Embedding API → 查询向量                          │
│  ② 查询向量 → 向量数据库 → 返回相似文档（数据库内部算相似度）  │
│  ③ 相似文档 + 问题 → LLM → 最终回答                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.5 你不需要重新造轮子的原因

生产环境的 RAG 技术栈已经非常成熟，你的代码只需要关心三件事：

1. **怎么把文档切分** → splitter 可以复用你现在的逻辑
2. **怎么调 Embedding API** → 可以复用你现在的代码
3. **怎么调向量数据库的 SDK** → 看官方文档，几行代码搞定

剩下的：
- ✅ 向量存储 → 交给向量数据库
- ✅ 相似度检索（余弦相似度等） → 交给向量数据库
- ✅ 持久化（重启不丢失） → 交给向量数据库
- ✅ 高性能索引（ANN 算法） → 交给向量数据库

**你现在自己写的这套代码，就是 LangChain 的"教学版"。学完这个，再看 LangChain 的 API，你会发现：哦，原来就是这么回事。**

---

## 九、总结

1. **RAG = 向量数据库 + 语义检索 + AI 生成**，解决的是"非结构化文本知识"的检索问题
2. **Document** 是 RAG 的基本数据单元，包含 `content`（文本）和 `metadata`（元数据）
3. **分割器**按分隔符优先级递归切分（段落→换行→句号→空格），优先在语义边界切分，兜底按字符数切分，相邻块有重叠避免上下文断裂
4. **Embedding** 把文本转成向量，底层是 HTTP 请求调用远程 AI 服务器的神经网络模型，返回的 base64 数据解码后得到浮点数列表
5. **检索**把查询也转成向量，用余弦相似度跟所有文档向量比对，返回最相似的 top-k 个
6. **工具机制**采用策略模式，每个工具一个类继承 `BaseTool`，通过 `name` 做字典查找，多态调用 `execute()`
7. **LLM 只负责选工具和传参数**，真正执行工具逻辑的是 Agent 代码
8. **RAG 只是工具箱里的一个选项**，和 SQL 查询、API 调用、联网搜索各司其职
9. **生产环境不用自己造轮子**，用 LangChain + 向量数据库 SDK，几行代码搞定。你现在学的原理，是为了以后能看懂和用好这些现成工具
