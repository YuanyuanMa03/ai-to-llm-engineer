# 01 · Tokenizer:BPE 与 SentencePiece

> 一句话:**学完这节,你能回答「BPE 怎么从字符一步步合并出子词」和「中文为什么不能按空格分词」这两个面试必考题。**

## 🤔 课前小测

先别急着学,花 30 秒回答下面 3 题(答案在文末):

1. 英文可以按空格切词,为什么 LLM 不直接全用 word 分词?
2. BPE 每一步合并的依据是什么——是「出现频率最高的 pair」还是「让 loss 最小的 pair」?
3. 词表里有 5 万个 token,输入里出现了一个词表外的生僻字组合,这叫什么问题?BPE 会怎么处理?

---

## 这节解决什么问题

模型看不懂文字,只看得懂数字。Tokenizer 就是文字和数字之间的翻译官:把一段文本切成一个个小单元(token),每个 token 对应词表里的一个整数 ID。这个切法看似不起眼,却直接决定了模型的序列长度、词表大小、多语言能力,甚至推理成本。

本节回答三个核心问题:为什么不能简单按 word/char 切?子词(subword)方案为什么是当前主流?BPE / WordPiece / SentencePiece 这三个名字到底差在哪?

## 核心概念 ★

### 为什么必须分词:三级方案的演进

文本无法直接喂给模型,必须先变成 ID 序列。切法有三级:

| 层级 | 例子(The lower) | 优点 | 致命缺点 |
|------|------------------|------|----------|
| **Word**(按词) | `["The", "lower"]` | 语义完整、序列短 | 词表爆炸;OOV 严重;形态学丢失(running≠run) |
| **Char**(按字符) | `["T","h","e"," ","l","o","w","e","r"]` | 词表极小(~百级);无 OOV | 序列太长;字符无语义;难学长距离依赖 |
| **Subword**(子词) | `["The","▁low","er"]` | 兼顾两者;高频词整块、低频词拆开 | 需要训练;切分不唯一 |

子词方案的核心思想:**高频整块,低频拆开**。「the」很常见就单独留一个 token;「lowerization」很罕见就拆成 `low + er + ization` 几个高频片段。这样既不爆词表,又没有 OOV。

### BPE:从字符开始的贪心合并

Byte-Pair Encoding(BPE)原是 1994 年的压缩算法,2015 年 Sennrich 把它引入 NLP 做分词。算法非常简单:

1. 把所有词拆成单个字符(作为初始「字符表」)。
2. 统计语料里所有相邻 token pair 的出现频率。
3. 把频率最高的 pair 合并成一个新 token,加入词表。
4. 重复 2-3,直到词表达到目标大小 $V$。

合并的频率统计公式:

$$
\text{score}(a, b) = \sum_{w \in \text{corpus}} \text{count}(w) \cdot \text{count}((a,b) \in w)
$$

即 pair $(a,b)$ 在整个语料中的总出现次数。每一步选 $\arg\max_{(a,b)} \text{score}(a,b)$ 合并。

BPE 训练完得到一张「合并规则表」,推理时按训练时的合并顺序贪心地应用到新文本。**词表大小 $V$ 与序列长度 $L$ 是反向关系**:

$$
L \approx \frac{\text{字符总数}}{\text{平均每个 token 覆盖的字符数}}, \quad V \uparrow \Rightarrow \text{平均覆盖} \uparrow \Rightarrow L \downarrow
$$

词表越大,序列越短(推理快、上下文窗口塞得多);但词表越大,Embedding 矩阵 $V \times d$ 越大,输出层 softmax 计算也越贵。常用区间是 **30k–100k**。

### WordPiece:BPE 的「似然」变体

WordPiece(BERT 用)和 BPE 几乎一样,区别只在**合并准则**。BPE 选频率最高的 pair;WordPiece 选让语料似然提升最大的 pair:

$$
\text{score}(a, b) = \frac{P(ab)}{P(a) \cdot P(b)}
$$

即合并后整体出现的概率,除以两个子串各自独立出现的概率之积。直觉:优先合并「在一起比分开更显著」的 pair(类似 PMI 点互信息)。

### SentencePiece:直接吃原始文本

BPE/WordPiece 通常需要**先按空格预分词**(英文友好)。SentencePiece(Google,2018)把整段原始文本(包括空格)当成一个 Unicode 字符流,直接在上面跑 BPE 或 Unigram 算法,空格本身用一个特殊符号 `▁`(U+2581)表示。

好处:**对中文/日文等无空格语言天然友好**,不需要预先分词;同一个模型可以同时处理中英日韩。LLaMA、T5、ALBERT 都用 SentencePiece。

另外 SentencePiece 还支持 **Unigram Language Model** 算法:先初始化一个大词表,用 EM 算法逐步删掉对似然贡献最小的 token,直到词表缩到目标大小。比 BPE 更概率化,切分可以多种(配合采样做数据增强)。

### tiktoken:GPT 系的快速 BPE

tiktoken(OpenAI)是 GPT-3.5/4 用的 BPE 实现,用 Rust 写核心、Python 调用,比 HuggingFace tokenizers 还快。关键特点:

- **直接在 UTF-8 字节**上做 BPE(byte-level BPE),彻底消灭 OOV——任何文本都能编码成字节序列。
- 词表大小:GPT-2 是 50257,cl100k_base(GPT-4)是 100277。
- 把空格也编进 token(如 `▁` → `Ġ`),不用 SentencePiece 那套特殊符号。

> ✅ **思考一下**:为什么 byte-level BPE 能彻底消灭 OOV?(提示:UTF-8 字节总数固定 256,任何文本都能落在这个表里,生僻字只是用更多字节表示而已。)

## 为什么这样设计

### 为什么子词赢了对 word/char

word 方案的 OOV 是死结:互联网上每天都有新词(人名、URL、错别字、代码变量),词表再大也盖不全;而且 `running / runs / ran` 学成三个独立 token,浪费参数。char 方案序列太长,「hello」要 5 个 token,模型很难学长依赖。子词是这两个极端的黄金分割。

### BPE vs WordPiece vs SentencePiece 三方对比

| 维度 | BPE | WordPiece | SentencePiece |
|------|-----|-----------|---------------|
| **合并准则** | pair 频率最高 | pair 似然增益最大 | 支持 BPE / Unigram 两种 |
| **是否需预分词** | 需要(按空格) | 需要(按空格) | **不需要**(吃原始文本) |
| **空格处理** | 当分隔符丢弃 | 当分隔符丢弃 | 用 `▁` 编码进 token |
| **OOV 处理** | 拆成已知子词或字节 | 拆成 `[UNK]` | 子词或字节,无 UNK |
| **代表模型** | GPT-2/4(tiktoken) | BERT、DistBERT | LLaMA、T5、ALBERT |
| **多语言友好度** | 英文友好,中文需先分词 | 同 BPE | **最友好**,原生支持中日韩 |
| **训练复杂度** | 低(贪心统计) | 低(似然统计) | Unigram 较高(EM 迭代) |

### 词表大小怎么选:三个权衡

| 因素 | 小词表(如 32k) | 大词表(如 100k) |
|------|------------------|-------------------|
| 序列长度 | 长(每 token 信息少) | 短(每 token 信息多) |
| 推理速度 | 慢(序列长) | 快(序列短) |
| Embedding 参数 | 少($32k \times d$) | 多($100k \times d$) |
| 多语言覆盖 | 差(小语种被拆碎) | 好 |
| 稀疏 token 学习 | 充分(高频) | 不足(很多 token 训不到) |

实践:英文为主 → 30k–50k;多语言 → 100k+;中文 LLaMA 衍生模型常用 65k–80k。

## 代码:最小实现

下面是一个**真能跑的极简 BPE**,在单词列表上学习合并规则,30 行纯 Python:

```python
import re
from collections import Counter

def get_pair_counts(corpus):
    """统计所有相邻 token pair 的频率。corpus 是 [(tokens, freq), ...]"""
    pairs = Counter()
    for tokens, freq in corpus:
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += freq
    return pairs

def merge_pair(pair, corpus):
    """在 corpus 上把指定 pair 合并成一个 token。"""
    new_corpus = []
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for tokens, freq in corpus:
        text = ' '.join(tokens)
        new_text = pattern.sub(''.join(pair), text)
        new_corpus.append((new_text.split(' '), freq))
    return new_corpus

# 训练语料:词 -> 频次(末尾 </w> 标记词边界)
vocab = {'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}
corpus = [(list(w) + ['</w>'], f) for w, f in vocab.items()]

# 学 10 次合并
merges = []
for _ in range(10):
    pairs = get_pair_counts(corpus)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    corpus = merge_pair(best, corpus)
    merges.append(best)

print("学到的合并规则(按顺序):")
for i, m in enumerate(merges, 1):
    print(f"  {i}. {m[0]} + {m[1]} -> {m[0]+m[1]}")
```

运行后你会看到 BPE 先合并 `e + s`(因为 newest/widest 共享),再合并 `est + </w>`,体现了「高频片段优先成词」。把它和 HuggingFace 的 `tokenizers.Tokenizer(models.BPE(...))` 对比,原理完全一致。

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/) 在线试跑,改 `vocab` 里的词和频次,看合并顺序怎么变。

## ⚠️ 易错点 / 面试陷阱

> ⚠️ **中文不按空格分词**。中文本身没有空格,直接按空格切会把一整句切成一个 token。LLaMA 用 SentencePiece 会把中文按 UTF-8 字节切,一个汉字常被拆成 2-3 个 token,所以中文 token 数显著多于英文。这也是中文 BPE 微调/扩词表项目的动机。

> ⚠️ **token ≠ word ≠ character**。`"Hello!"` 可能是 `["Hello", "!"]` 两个 token;`"don't"` 可能是 `["don", "'t"]`。算 token 数要用 `len(tokenizer.encode(text))`,不能按空格数单词。

> ⚠️ **词表大小不是越大越好**。词表过大 → Embedding 矩阵膨胀、稀疏 token 训练不充分(很多 token 在语料里只出现几次,Embedding 学不好)。GPT-2 用 50k,GPT-4 升到 100k 是因为加了多语言,不是「越大越先进」。

> ⚠️ **BPE 推理是确定性的贪心匹配,按训练时的合并顺序**。所以同样词表 + 同样语料,BPE 切分结果唯一;但 Unigram(SentencePiece)可以有多种切分,支持采样。

## 🎯 面试会怎么考

- **八股题**:讲一下 BPE 的训练流程;为什么中文不能用空格分词;WordPiece 和 BPE 的合并准则差在哪;OOV 是什么,BPE 怎么解决的。
- **手撕题**:给一个语料和一个 pair,写出 BPE 合并一步后的 corpus;实现 `get_pair_counts`(就是上面代码里的统计函数)。
- **深挖题**:词表大小一般定多少,为什么?BPE 和 Unigram 算法本质区别(贪心 vs 概率删减)?为什么 GPT 用 byte-level BPE 而不是 SentencePiece?tiktoken 比 HuggingFace 快在哪?

## 📂 简历可写的项目

**项目名:中文 BPE 分词器训练与对比实验**

描述:基于 HuggingFace `tokenizers` 库,在维基百科中文语料(约 100MB)上训练 4 个不同词表大小(16k / 32k / 65k / 100k)的 BPE 分词器,对比同一批测试文本在 token 数量、平均 token 覆盖字符数、OOV 率上的差异;并对比 SentencePiece Unigram 在中英混排文本上的表现。输出一份分析报告,给出「中文场景推荐词表大小」的结论。

技术栈:Python、HuggingFace tokenizers、sentencepiece、matplotlib、pandas。

亮点词:**自训练分词器、词表消融实验、中英混排处理、token 效率分析**。

## 🚀 挑战

用 `tiktoken` 分别对同一段中英对照文本(各 1000 字/词)用 `gpt2`(50257)和 `cl100k_base`(GPT-4,100277)编码,统计 token 数。你会发现:

1. 同一段中文,`cl100k_base` 比 `gpt2` 省多少 token?(提示:GPT-4 词表扩了中文)
2. 估算:用 GPT-2 处理 100 万字中文小说,要花多少 token?按 GPT-4 API $0.01/1k token 算,光分词差异就值多少钱?

把结果写成一条推文长度的心得。

## 🔗 延伸阅读

1. Sennrich et al., 2016, *Neural Machine Translation of Rare Words with Subword Units* —— BPE 引入 NLP 的原始论文,2 页算法描述,必读。
2. HuggingFace NLP Course, Chapter 2: Tokenizers —— 配合 `tokenizers` 库实操,有交互式 demo。
3. OpenAI, tiktoken GitHub README —— 直接看代码示例,理解 byte-level BPE 的 `bytes_to_unicode` 映射。

---

## ✅ 课后小测(答案)

**课前小测答案:**

1. **Word 分词有三个死结**:① 词表爆炸(组合爆炸,几十万词);② OOV 严重(新词、人名、URL、错别字都无法处理);③ 形态学信息丢失(`running/runs/ran` 学成独立 token,参数浪费且语义不共享)。子词方案「高频整块、低频拆开」同时解决了这三个问题。

2. **BPE 合并的依据是「出现频率最高的 pair」**,不是 loss。每一步统计语料里所有相邻 token pair 的频次,选最大的合并。WordPiece 才是用「似然增益」$P(ab)/(P(a)P(b))$ 作为准则——这是两者最核心的区别。

3. **这叫 OOV(Out-Of-Vocabulary)问题**。BPE 的处理:把生僻组合拆成已知的子词或字节。如果是 byte-level BPE(如 GPT),任何文本都能落到 256 个字节上,**彻底没有 OOV**;如果是传统 BPE,最坏情况拆到字符级,也能表示,只是 token 数变多。WordPiece/BERT 用 `[UNK]` 兜底,信息就丢了。

**掌握自检:** 给定语料 `{low:5, lower:2, newest:6, widest:3}`,手算 BPE 前 3 步合并分别合并了什么 pair、合并依据的频率各是多少。

> 参考答案:第 1 步,所有 `e + s`(newest 出现 1 次×6 + widest 出现 1 次×3 = 9 次)和 `e + r`(low→lower 无,lower 1 次×2 + ... 实际要看词边界),把所有 pair 频率列出来取最大;关键是理解「频次 = 词频 × pair 在该词内出现次数」。能写对这个统计过程,BPE 就通了。
