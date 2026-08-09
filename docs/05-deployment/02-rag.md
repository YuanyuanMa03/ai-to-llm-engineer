# 02 · RAG:检索增强生成

> 一句话:**学完这节,你能回答「RAG 全流程是什么、为什么要 rerank、向量检索用余弦还是欧氏」这个面试题。**

## 🤔 课前小测
先别急着学,花 30 秒回答下面 3 题(答案在文末):
1. 模型知识过时了,为什么不能用"重新微调一下"了事,非得上 RAG?
2. 向量检索里,大家默认用余弦相似度而不是欧氏距离,直觉上为什么?
3. 文档切 chunk 时,chunk 越大召回越准、还是越小越准?为什么这是个错的问题?

---

## 这节解决什么问题

大模型有三个根上的毛病:**知识过时**(训练截止后的事它不知道)、**幻觉**(把没发生过的事编得头头是道)、**不懂私有数据**(你公司的内部文档它没见过)。这三个毛病靠纯 prompt 都治不好。

解法两条路:**微调**——把新知识塞进权重;**RAG**——推理时临时检索外部知识塞进 prompt。微调贵(数据 + 算力)、慢(每次更新都要训)、知识更新一次要重训一次;RAG 廉价、可热更新、能溯源、能做权限控制。所以现实里几乎所有"知识密集型应用"都是 RAG,而不是微调。这一节讲清 RAG 的全流程、易踩的坑,以及为什么"朴素 RAG"经常翻车。

## 核心概念 ★

### 1. RAG 的全流程

一条 RAG 链路分**离线索引**和**在线检索 + 生成**两段:

```
[离线:建索引]
文档 → 切 chunk → embedding 模型编码 → 向量 → 存入向量库

[在线:问答]
用户问题 → embedding 编码 → 向量库检索 top-k → (可选) rerank 精排
       → 把 top-k chunk 拼进 prompt → LLM 生成答案
```

五个核心动作:**chunk、embed、retrieve、rerank、generate**。RAG 工程师 80% 的时间花在前四步——生成那一步反而最稳。

### 2. Embedding 模型:把文本变成向量

Embedding 模型(BGE、E5、text-embedding-3、Cohere embed)把任意长度文本压成一个固定维度的向量(比如 768 或 1536 维)。**核心约束:语义相似的文本,向量在空间里也相近**。

训练方式主流是**对比学习**:正样本(同义句)拉近、负样本(无关句)推远。损失函数常见 InfoNCE:

$$
\mathcal{L}=-\log\frac{\exp(\text{sim}(q,x^+)/\tau)}{\exp(\text{sim}(q,x^+)/\tau)+\sum_{x^-}\exp(\text{sim}(q,x^-)/\tau)}
$$

其中 $\text{sim}$ 是余弦相似度,$\tau$ 是温度。直觉:**让正样本在所有候选里相似度最高**。

> ⚠️ Embedding 模型和 LLM 是**两回事**。Embedding 输出向量(给检索用),LLM 输出 token(给生成用)。RAG 里两者各管一段,不能混。常见新手错误:用 LLM 自己的 hidden state 当 embedding——可以,但效果远不如专门训练的 embedding 模型。

### 3. 向量检索:余弦相似度

检索本质是**在向量库里找和查询向量最近的 k 个**。距离度量最常用**余弦相似度**:

$$
\cos(a,b)=\frac{a\cdot b}{|a|\,|b|}\in[-1,1]
$$

值越大(越接近 1)越相似。为什么不用欧氏距离?因为 embedding 模型训练时几乎都**按方向**优化,向量模长意义不大,余弦只看方向夹角,更鲁棒。欧氏距离会被向量模长干扰——模长大的文档"看起来远",其实只是写得更长。

工程上向量库通常要求**入库前先归一化**($a \leftarrow a/|a|$),这样余弦相似度退化为点积 $a\cdot b$,**算得更快**(省掉两次开方)。

$$
\text{归一化后:}\quad \cos(a,b)=a\cdot b
$$

### 4. 向量库(FAISS / Milvus / Chroma)

向量库解决**海量向量的近似最近邻(ANN)检索**。暴力点是 $O(N\cdot d)$,千万级以上不可行。ANN 用 HNSW、IVF、PQ 等索引,**牺牲一点精度换百倍速度**。

| 向量库 | 定位 | 适合场景 |
|---|---|---|
| **FAISS** | 库(Meta 出),不是服务 | 单机、原型、研究 |
| **Milvus** | 分布式向量数据库 | 生产、亿级向量、高可用 |
| **Chroma** | 轻量嵌入式 | 原型、demo、小数据 |
| **pgvector** | Postgres 扩展 | 已有 Postgres、不想多引一个组件 |
| **Qdrant** | Rust 写的向量数据库 | 高性能、过滤检索强 |

### 5. Rerank:重排序

向量检索召回的 top-k 里常有"看起来方向接近、其实不解决问题"的噪声。**Rerank 用一个更重(但更准)的模型**对这 k 个候选用 query 重新打分,挑出真正相关的几个。

- 检索:bi-encoder(query 和 doc 独立编码 → 点积),快但糙。
- Rerank:cross-encoder(query 和 doc 拼一起喂给模型 → 输出一个相关性分),慢但准。

> ✅ **思考一下**:为什么 rerank 不直接代替检索?提示:cross-encoder 要把每个 doc 都和 query 喂一遍大模型,在**百万级文档**上做这件事会怎样?这就是 RAG 标准的"先粗筛后精排"两段式——和搜索引擎一个道理。

### 6. Advanced RAG:朴素 RAG 不够时

朴素 RAG 在简单问答上够用,复杂场景会塌。常见增强:

| 技巧 | 解决什么 | 例子 |
|---|---|---|
| **查询改写** | 用户问得烂,直接检索效果差 | "它多少钱" → 用历史上下文补全成"iPhone 15 多少钱" |
| **多路召回(Hybrid)** | 单一向量召回漏召回 | 向量检索 + BM25 关键词检索,取并集再 rerank |
| **多跳检索** | 答案分散在多个文档 | 先查"X 是谁",再拿结果查"X 的公司是什么" |
| **自反思(Self-RAG)** | 模型不确定时主动重检索 | 模型输出前自评"检索结果够不够",不够就再查一次 |
| **Parent-child chunking** | 小 chunk 检索、大 chunk 喂模型 | 用小段精确匹配,但把整段父文档塞给 LLM 保留上下文 |

### Chunk 策略:RAG 调参的第一个杠杆

chunk 怎么切,直接决定召回质量。四种主流策略,各有适用场景:

| 策略 | 怎么切 | 优点 | 缺点 |
|---|---|---|---|
| **固定长度** | 每 N 个 token 一刀,带 overlap | 实现最简单 | 容易切断句子 / 表格 |
| **递归字符**(RecursiveCharacterTextSplitter) | 按 `\n\n → \n → 。 → 空格` 优先级切,尽量在自然边界断 | 兼顾长度和语义,LangChain 默认 | 长表格 / 代码仍可能乱 |
| **语义切**(SemanticChunker) | 用 embedding 相邻句相似度,相似度突降处断 | chunk 内语义最纯 | 慢,chunk 长度不可控 |
| **文档结构感知** | 按 Markdown 标题 / HTML 标签 / PDF 段落切 | 保留原文档逻辑层级 | 依赖文档格式规整 |

生产里的常见组合是**「递归字符 + parent-child」**:用递归切成 ~300 token 的子 chunk 做检索(精确匹配),但喂给 LLM 时把子 chunk 所属的父 chunk(整节 / 整段)塞进去(保留上下文)。这样既检索准,又不会让模型看到半句话。

一个常被忽略的细节:**表格和代码不要用文本 splitter 切**。一个 10 行表格被从中间切开,两边都失去意义。正确做法是解析时把表格作为一个独立 chunk,代码同理。

---

## 为什么这样设计

### 微调 vs RAG

| 维度 | 微调(往权重里塞知识) | RAG(推理时检索) |
|---|---|---|
| 知识更新成本 | 高,每次都要重训 | 低,改文档即可,热更新 |
| 幻觉率 | 高(模型凭记忆答,容易编) | 低(有据可循,可溯源) |
| 私有数据 | 训练数据混入权重,**难做权限** | 文档可分库分权限,**天然隔离** |
| 实时信息 | 不可能(权重是静态的) | 完全可以(接实时数据源) |
| 适合场景 | 调风格、改能力、领域适配 | 知识问答、文档检索、企业知识库 |
| 代价 | 数据 + 算力 + 周期 | 工程成本,几乎无训练 |

结论:**"改能力"用微调,"加知识"用 RAG**,生产里经常两者结合(RAG 答知识 + 微调调风格)。

### 朴素 RAG vs Advanced RAG

| 维度 | 朴素 RAG | Advanced RAG |
|---|---|---|
| 检索策略 | 单路向量 top-k | 多路召回(Hybrid)+ rerank |
| 查询处理 | 用户原话直接 embed | 改写 / 分解 / HyDE(假设性文档) |
| chunk 策略 | 固定长度切 | 语义切 / parent-child / 滑窗 |
| 失败处理 | 一次性,好坏听天 | 自反思,检索不够再来一轮 |
| 效果天花板 | 中等 | 显著提升(常 +10~20%) |

朴素 RAG 是 80 分方案,Advanced RAG 才是生产可用。但**先做朴素版跑通,再针对瓶颈优化**,不要一上来就堆技巧。

### 检索和生成,谁决定 RAG 的天花板

一个常见误区是"RAG 不准就换更大的 LLM"。实测下来,RAG 链路的瓶颈**绝大多数时候在检索,不在生成**。把整个链路拆开看贡献:

| 环节 | 典型问题 | 对最终效果的影响 |
|---|---|---|
| chunk 切分 | 关键信息被切断 / 一个 chunk 多主题 | ★★★(召回源头就错了) |
| embedding 模型 | 领域不匹配(中文用英文模型) | ★★★(整个向量空间是错的) |
| 检索召回 | top-k 太小 / 单路召回漏召回 | ★★★(根本没把答案召回进来) |
| rerank | 没有 / 模型选错 | ★★(召回进来了但排到 k 外) |
| 生成 LLM | 太小 / prompt 没设计"无答案"兜底 | ★(只要上下文对,小模型也够) |

记住一条排序:**先 chunk → 再 embedding → 再检索 → 再 rerank → 最后才轮到生成 LLM**。优化顺序反了,花大钱换大模型却涨不动点。

### RAG 的"无答案"判定比"能回答"更难

朴素 RAG 还有个工程盲区:**当检索结果和问题无关时,模型应该回答"不知道",而不是强行用无关上下文编一个答案**。这件事 prompt 工程能做一部分(明确写"资料里没有就回答不知道"),但模型经常"忍不住"用无关片段硬答。生产做法:

- 检索后先算 query 和 top-1 chunk 的相似度,**低于阈值直接走"无答案"分支**,根本不喂 LLM。
- 让 LLM 输出结构化结果(`{"has_answer": bool, "answer": str, "citations": [...]}`),后端按 `has_answer` 分流。
- 高要求场景再加一个**答案一致性校验**:让另一个 LLM 检查"答案是否真的被引用的 chunk 支持",不支持就打回。

这套"防御性 RAG"是 ToB 项目的标配,开源 demo 里几乎没人讲,但面试官问"你的 RAG 怎么避免幻觉"时,这就是区分度。

## 代码:最小实现

下面用 numpy 写一个**极简 RAG**:3 个文档 chunk → 用随机向量模拟 embedding → 查询 → top-1 检索 → 拼 prompt。**不依赖任何框架**,看清流程本质。

```python
import numpy as np

# === 1. 模拟 embedding 函数(真实场景用 BGE / OpenAI text-embedding-3) ===
# 用文本 hash 喂伪随机数生成器,把任意文本变成 64 维向量。
# 注意:Python 字符串 hash 默认每次进程启动会随机化(PYTHONHASHSEED),
#       所以不同运行里向量会变,排序也可能变——这只是教学示意,
#       真实 embedding 模型是固定的神经网络,同一文本永远得同一向量。
def fake_embed(text, dim=64):
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)   # 归一化!入库前必做

# === 2. 离线:3 个 chunk 建索引 ===
chunks = [
    "Llama 是 Meta 开源的大语言模型家族,2023 年发布 Llama-2。",
    "vLLM 用 PagedAttention 优化 KV Cache,推理吞吐提升数倍。",
    "Transformer 的核心是自注意力机制,公式为 softmax(QK^T/sqrt(d)) V。",
]
chunk_embs = np.stack([fake_embed(c) for c in chunks])   # (3, 64)

# === 3. 在线:检索 ===
query = "vLLM 怎么做推理加速?"
q_emb = fake_embed(query)                                 # (64,)
sims = chunk_embs @ q_emb                                 # 点积 = 余弦(已归一化)
top_k = 3
top_idx = np.argsort(-sims)[:top_k]
print("召回排序:", [(i, round(float(sims[i]),3)) for i in top_idx])
# 期望:vLLM 那条相似度最高
```

```python
# === 4. 拼 prompt 喂 LLM(这里用打印代替真实生成) ===
retrieved = "\n".join([f"[{i}] {chunks[i]}" for i in top_idx])
prompt = f"""你是一个严谨的助手,根据下面的资料回答问题。资料里没有就回答"不知道"。

【资料】
{retrieved}

【问题】{query}
"""
print(prompt)
# 真实场景:response = llm.chat(prompt)
```

跑一下,你会看到召回排序(由于上面提到的 hash 随机化,具体顺序每次进程可能不同;真实 embedding 模型下,vLLM 那条会稳定排第一)。这就是 RAG 的最朴素形态。生产里把 `fake_embed` 换成真 embedding 模型、把 `chunk_embs` 换成 Milvus/FAISS 检索、把最后那段 prompt 喂给真 LLM,就是可用的 RAG。

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/) 在线试跑,把 query 换成"Transformer 的注意力公式",看召回排序怎么变。

补一段 **top-k 检索 + rerank** 的极简示意:

```python
# 假装 cross-encoder:把 query 和每个 doc 拼一起,用一个更"重"的打分函数
def fake_rerank_score(query, doc):
    # 真实里是 cross-encoder 模型输出,这里用字符重叠粗略模拟"精排"
    overlap = len(set(query) & set(doc))
    return overlap

# 第一步:向量检索召回 top-5(粗)
recall_k = min(5, len(chunks))
candidates = np.argsort(-sims)[:recall_k]
# 第二步:rerank 精排,挑 top-2
reranked = sorted(candidates, key=lambda i: -fake_rerank_score(query, chunks[i]))
final = reranked[:2]
print("rerank 后:", final)
```

## ⚠️ 易错点 / 面试陷阱

> **chunk 切太大,召回不准;切太小,丢上下文。** 切 1000 token 时一个 chunk 塞了多个主题,检索向量被"稀释",top-k 可能全是跑题的;切 50 token 时语义残缺,模型看到半句话答不全。**没有"最优 chunk 大小"**,要按文档类型实测:问答类文档 200-400 token,代码 / 长文 500-800,再叠 overlap(滑窗)防边界割裂。

> **embedding 模型和 LLM 不是一回事,且不能用错。** embedding 只输出向量(给检索),LLM 输出 token(给生成)。常见错:用 `gpt-3.5-turbo` 当 embedding(它不输出 embedding 接口);或用 OpenAI 的 embedding 去检索中文 BGE 训练的语料(领域不匹配,召回差)。

> **rerank 不是可选项,是 RAG 效果的最大杠杆之一。** 很多团队朴素 RAG 上线后回答不准,第一反应是"换更大的 LLM",其实加一个 BGE-reranker / Cohere rerank 常常比换模型涨点更多、成本却低一个数量级。

> **多路召回(Hybrid)比纯向量召回稳。** 纯向量对**专有名词、代码、人名**这类精确匹配不敏感(它看的是语义方向)。叠一个 BM25 关键词检索,把"提到 iPhone 15 Pro Max 的文档"用关键词精确捞回来,再用向量补语义相关,准确率提升明显。

> **中文场景要选中文 embedding 模型。** 用英文模型 embed 中文,向量空间分布完全不对,召回会一塌糊涂。中文选 BGE-zh、M3E、bge-large-zh;多语言用 BGE-M3 / Cohere multilingual。

> **没有评测集,RAG 调参全是玄学。** 很多团队调 chunk 大小、换 embedding、加 rerank 全靠"感觉变好了",其实没法量化。正确做法:**先建 50-200 题的评测集**(问题 + 标准答案 + 答案所在文档),再分两步评:**检索阶段**用 Recall@k(标准文档是否进了 top-k)、MRR;**端到端**用人工标注的准确率或 LLM-as-judge。有这两个数,每次改动才知道是涨是跌。

> **LLM-as-judge 便宜但有偏。** 用 GPT-4 给 RAG 答案打分省人力,但大模型评判会偏向"长得像标准答案"的输出,对风格不同的回答打分不稳。重要评测仍要抽 20% 人工复核,别全信 judge 模型。

## 🎯 面试会怎么考

- **八股题**:RAG 全流程?为什么需要 rerank,它和检索的区别是什么?向量库怎么选?chunk 大小怎么定?
- **手撕题**:写一个余弦相似度函数(注意归一化);或写一个 top-k 检索(给定 query 向量和向量矩阵,返回 top-k 索引)。
- **深挖题**:RAG 怎么解决多跳问答?Hybrid(向量 + BM25)为什么比纯向量好?Self-RAG 的反思机制是什么?如果检索结果完全无关,模型怎么避免被误导(prompt 怎么设计、有没有"无答案"判定)?

## 📂 简历可写的项目

**项目名:基于私有 PDF 的文档问答系统(带 rerank 的 RAG)**

- **描述**:搭一个端到端 RAG:用 PyMuPDF 解析 PDF → RecursiveCharacterTextSplitter 切 chunk(带 overlap) → BGE-zh 编码 → 存 Milvus / FAISS → 检索 top-10 → BGE-reranker 精排 top-3 → 喂 LLM 生成答案,带**原文引用溯源**。
- **技术栈**:LangChain / LlamaIndex、BGE embedding + reranker、Milvus / FAISS、任意开源/闭源 LLM。
- **加分点**:
  - 建一个 50-100 题的评测集,对比「无 RAG / 朴素 RAG / 加 rerank / Hybrid」四档准确率,画出提升曲线。
  - 实现**parent-child chunking**:小 chunk 检索、父 chunk 喂模型,处理"答案在表格 / 跨段落"场景。
  - 接入查询改写,把"它怎么样"这种代词性问题补全后再检索。

## 🚀 挑战

**做一个对照实验**:用 30 段技术文档(随便从官方文档爬),建两套 RAG:

1. **朴素 RAG**:固定 chunk 500 token,向量检索 top-5,直接生成。
2. **加 rerank**:向量检索 top-20 → rerank 挑 top-3 → 生成。

各自准备 10 个问题,人工标注答案是否正确、是否引用了相关段落。**对比两组的准确率**,感受 rerank 的提升有多大。

进阶:再试一组 **Hybrid(向量 + BM25)**,看看对"包含专有名词的问题"提升是否更明显。

## 🔗 延伸阅读

1. 📄 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks(Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) —— RAG 的开山论文,首次提出把检索器和生成器联合训练的框架,术语的来源。
2. 📄 [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection(2023)](https://arxiv.org/abs/2310.11511) —— Advanced RAG 的代表作,模型自己决定"要不要检索 / 检索结果够不够",把 RAG 从硬编码流程变成可学习策略。
3. 💻 [LangChain RAG 教程 + LlamaIndex 文档](https://docs.llamaindex.ai/) —— 工程上最常踩的两套框架,LlamaIndex 对 RAG 抽象更清晰、LangChain 生态更全;两者任选一个跑通官方 quickstart,就能动手做项目。

---

## ✅ 课后小测(答案)

**课前小测答案:**
1. **微调治不了知识过时**:每次更新知识都要重训(贵 + 慢),且知识混进权重,无法溯源、无法做权限隔离、无法撤回。RAG 把知识放在外部知识库,改文档即生效,天然适合频繁更新、需要审计的企业场景。
2. **余弦只看方向(语义夹角),欧氏会被向量模长干扰。** Embedding 模型按方向优化,模长意义不大;长文档模长大、欧氏距离"看起来远",但其实只是写得多。归一化后余弦 = 点积,还更快。
3. **这是个错的问题——chunk 大小没有"越大越准"或"越小越准",是个 trade-off。** 大 chunk 召回方向被稀释(一个 chunk 多主题),小 chunk 丢上下文(语义残缺)。要按文档类型实测,常用 200-800 token + overlap,再用 parent-child / 语义切分等技巧缓解。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——

> 用户问"我们公司 Q3 的销售额是多少",公司内部有一个 200 页的财报 PDF。请你说出这条查询**从用户敲回车到拿到答案**经过的每一步,以及每一步可能出问题的地方。
>
> **答**:① 查询处理(可改写、补上下文)→ ② embedding 编码 query → ③ 向量库检索 top-k(可叠 BM25 多路召回)→ ④ rerank 精排 → ⑤ 把 top-k chunk 拼 prompt → ⑥ LLM 生成答案(带引用)。每步的坑:①代词未消解 ②embedding 模型对中文 / 数字不敏感 ③chunk 切割把"Q3 销售额"和它的表格割开 ④没 rerank,噪声 chunk 进了 top-3 ⑤prompt 没设计"无答案"判定,模型强行编 ⑥LLM 把检索到的"Q3 营收"和"Q3 利润"搞混。生产 RAG 的工程量,大半都在堵这些缝。
>
> **额外要点**:这个例子里"Q3 销售额"在 PDF 里很可能是一张表格的某个单元格。如果用普通文本 splitter,表格会被切碎,检索根本召不回来——这正是"文档结构感知切分"和"parent-child chunking"要解决的:把整张表格作为一个独立 chunk,检索命中后把表格所属章节一起喂给 LLM。能在自检答案里主动点出"表格不能乱切",说明你真的踩过 RAG 的坑。
