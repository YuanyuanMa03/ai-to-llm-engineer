# 03 · Transformer 结构详解

> **这一关结束，你能手写 Attention，并讲清为什么要除以 $\sqrt{d_k}$。**

## 🤔 课前小测
先别急着学,花 30 秒回答下面 3 题(答案在文末):
1. Attention 公式里 $QK^T$ 之后为什么要除以 $\sqrt{d_k}$ 再 softmax?不除会怎样?
2. Multi-Head Attention 把 $d_{model}$ 维向量切成多个头,为什么不直接用一个大单头?
3. Causal Mask 是上三角还是下三角?它屏蔽的是"过去 token"还是"未来 token"?

---

## 这节解决什么问题

Transformer 是当今所有大语言模型(GPT、LLaMA、Qwen、Claude……)的共同骨架。从 2017 年《Attention is All You Need》提出至今,Transformer 已经统治了 NLP、CV、语音、强化学习几乎所有模态。可以说,**不理解 Transformer,就没法读任何一篇现代大模型论文,也没法通过任何一场大模型算法岗面试**。

这一节我们把它拆到最小零件:Self-Attention、Multi-Head、Position Encoding、残差与 LayerNorm、FeedForward、Encoder/Decoder 的差异。每一块都告诉你**为什么这么设计**,而不是背诵"加 LayerNorm、用残差、四层堆叠"。

## 核心概念 ★

### 1. Self-Attention 的直觉:数据库检索

把 Attention 想象成一次**软性数据库查询**:
- $Q$(Query):我想查什么(检索词)。
- $K$(Key):数据库里每条记录的"标签"(用于被检索匹配)。
- $V$(Value):数据库里每条记录的"实际内容"。

Attention 用 $Q$ 和每个 $K$ 算相似度,softmax 得到一组权重(对全部 key 的注意力分配),再加权求和 $V$,得到最终输出。**Self**-Attention 是指 $Q/K/V$ 都来自同一序列,即序列里每个位置都去"检索"自己序列里的所有位置。

### 2. Scaled Dot-Product Attention(灵魂公式)

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$:每个 Query 对每个 Key 的点积相似度,$(N, d_k)\times(d_k, N)\rightarrow(N, N)$。
- $\sqrt{d_k}$:缩放因子,**防止点积数值过大导致 softmax 饱和**(下面单独讲)。
- $\text{softmax}$:把每一行(每个 Query 对所有 Key 的相似度)归一化成注意力权重。
- $\times V$:用权重加权聚合 Value,得到 $(N, d_v)$ 输出。

### 3. 为什么除以 $\sqrt{d_k}$ ★

设 $Q, K$ 的每个分量独立、均值 0、方差 1,则点积 $Q\cdot K=\sum_{i=1}^{d_k}Q_iK_i$ 的方差是 $d_k$。当 $d_k$ 大(如 64)时,点积数值会很大,softmax 会进入**梯度接近 0 的饱和区**:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}},\quad \frac{\partial}{\partial z_i}\text{softmax}_i = \text{softmax}_i(1-\text{softmax}_i)$$

最大 logits 远大于其他 → softmax 输出近似 one-hot → 梯度几乎为 0 → 训练停滞。除以 $\sqrt{d_k}$ 把方差拉回 1,让 softmax 留在工作区。这是个纯数值稳定性的工程技巧,但**面试高频到不行**。

### 4. Multi-Head Attention:多子空间并行

单头 attention 只能学一种"关注模式"(例如主谓一致)。Multi-Head 把 $d_{model}$ 切成 $h$ 个头,每个头维度 $d_k=d_{model}/h$,独立做 attention,再 concat 后投影:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

直觉:不同头能学到**不同类型的关系**(语法、共指、语义相似、位置远近……),类似 CNN 用多个卷积核学不同特征。注意总参数量与单头基本相同(只是把 $d$ 切了),所以多头是"免费的多样性"。

### 5. 位置编码:Attention 不知道顺序

Attention 对输入是**集合操作**(每个位置直接看所有位置),本身完全无序。把"我 打 你"和"你 打 我"喂进去,attention 输出会完全一样。所以必须额外注入位置信息。

- **Sinusoidal(原论文)**:用不同频率的 sin/cos 给每个位置一个确定性的 $d_{model}$ 维向量。
  $$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$$
  $$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$$
- **可学习位置编码(BERT/GPT-2)**:位置嵌入当作普通参数训出来。
- **RoPE(旋转位置编码,LLaMA/GPT-NeoX)**:把位置信息以旋转矩阵形式注入 $Q$、$K$,使得**相对位置** $q_m^T k_n$ 只依赖 $m-n$,外推性好,是目前大模型主流。

### 6. 残差连接 + LayerNorm(Add & Norm)

每个子层外面套:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

- **残差**:让梯度能"短路"流过深网络,缓解梯度消失。没了它 Transformer 根本堆不深。
- **LayerNorm**:对每个样本、每个 token 内部做归一化(均值 0、方差 1)。和 BatchNorm 不同,**LN 不依赖 batch 维**,所以 batch=1 推理也能用,且不受序列长度影响。

### 7. FeedForward 层(MLP)

每个位置独立过一个两层 MLP:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

通常中间维度是 $d_{model}$ 的 4 倍。Attention 负责"信息路由"(决定哪些位置互相看),FFN 负责"信息变换"(把学到的关联映射到新表示)。**两者分工,缺一不可**。

### 8. Encoder vs Decoder

- **Encoder(BERT)**:Self-Attention 是双向的,每个位置能看到全部位置。适合**理解类**任务(分类、抽取)。
- **Decoder(GPT)**:Self-Attention 加 **Causal Mask**(上三角置 $-\infty$),每个位置只能看到自己和过去 → 自回归生成。底层还有一路 Cross-Attention(原论文 Decoder)从 Encoder 输出取 K/V。
- **Encoder-Decoder(T5/BART)**:Encoder 双向编码源序列,Decoder 自回归生成目标,中间用 Cross-Attention 桥接。

> ✅ **思考一下**:BERT 是 Encoder-only、GPT 是 Decoder-only,为什么"生成式大模型"最终都选了 Decoder-only?提示:训练目标不同(掩码语言模型 vs 下一 token 预测),以及 scaling law 上的差异。

## 为什么这样设计

### 三种主流 Transformer 架构对比

| 架构 | 代表模型 | Attention | 训练目标 | 适合任务 |
|---|---|---|---|---|
| Encoder-only | BERT、RoBERTa | 双向 Self-Attn | 掩码语言模型(MLM) | 分类、抽取、嵌入 |
| Decoder-only | GPT、LLaMA、Qwen | 因果 Self-Attn(Causal Mask) | 下一 token 预测 | 生成、对话、Few-shot |
| Encoder-Decoder | T5、BART | Encoder 双向 + Decoder Cross-Attn | Span 损坏 + 重构 | 翻译、摘要 |

工程上的趋势:**大模型主流是 Decoder-only**。原因:训练目标统一(下一个 token)、数据效率高(每个位置都贡献一次 loss)、scaling 友好、Few-shot 能力强。Encoder-Decoder 在翻译等 seq2seq 场景仍有市场。

## 代码:最小实现

用 numpy 实现 2×2 的 Scaled Dot-Product Attention(纯矩阵运算,~15 行):

```python
import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x = x - np.max(x, axis=axis, keepdims=True)   # 减最大值防溢出
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (N, d_k) 形状,N=序列长度
    mask:    可选,(N, N) 形状,要屏蔽的位置填 -inf
    """
    d_k = Q.shape[-1]
    # 1) 相似度:QK^T,形状 (N, N)
    scores = Q @ K.T / np.sqrt(d_k)
    # 2) 可选:加 mask(Decoder 的 Causal Mask 用)
    if mask is not None:
        scores = scores + mask
    # 3) softmax 得到注意力权重,每行加起来=1
    weights = softmax(scores, axis=-1)
    # 4) 加权求和 V,输出形状 (N, d_v)
    return weights @ V, weights

# 跑一个 2 token、d_k=2 的例子
Q = np.array([[1.0, 0.0],   # token1 的 query
              [0.0, 1.0]])  # token2 的 query
K = V = Q                    # Self-Attention:K、V 同源
out, w = scaled_dot_product_attention(Q, K, V)
print("attention weights:\n", w)
print("output:\n", out)
```

再加一个 Causal Mask 验证屏蔽效果:

```python
# Causal Mask:上三角(未来 token)置 -inf,只允许看自己和过去
N = 2
mask = np.triu(np.full((N, N), -np.inf), k=1)   # [[0, -inf],[0, 0]]
out_c, w_c = scaled_dot_product_attention(Q, K, V, mask=mask)
print("causal weights:\n", w_c)
# token1 看不到 token2,第一行 = [1.0, 0.0]
```

> 💡 点「运行到这里」在线试跑。把 `d_k` 改成 512,再去掉 `/np.sqrt(d_k)`,打印 `weights`,你会看到 softmax 退化成 one-hot —— 直观感受"为什么要缩放"。

## ⚠️ 易错点 / 面试陷阱

> **Causal Mask 是上三角屏蔽"未来 token"。** 上三角(对角线及以上,`k=1`)填 $-\infty$,softmax 后权重变 0。常见错误:记成"下三角""屏蔽过去"。一句话记忆:**保留"看自己和过去"的下三角,屏蔽"未来"的上三角**。

> **除以 $\sqrt{d_k}$ 是为了数值稳定,不是数学必然。** 有的论文除 $\sqrt{d_k}$、有的除 $d$(如某些线性 attention 变体)、有的不除但加温度参数。原论文的论证是"假设 $Q,K$ 各分量独立均值 0 方差 1,点积方差为 $d_k$"——**这是个分析,不是定理**。面试答到"防止 softmax 饱和、梯度消失"就够。

> **位置编码是必须的,不是可选优化。** Attention 本身是集合操作、对顺序不敏感,不注入位置信息,模型连"主语在前、谓语在后"都分不清。RoPE 的精髓是把"绝对位置"以旋转矩阵注入,使最终点积只依赖**相对位置** $m-n$,因此外推到训练时没见过的长度效果更好——这是当前主流大模型几乎都用 RoPE 的原因。

> **Multi-Head 的总参数量 ≈ 单头。** 切成 $h$ 个头,每个头 $d/h$ 维,权重矩阵总大小 $d\times d$ 不变。多头带来的是"表示的多样性",不是参数量增加。面试有人答"多头是为了增加参数"是错的。

> **残差连接不是"锦上添花"。** 没有 Add,Transformer 堆到 12 层以上就训不动了。它的作用有两层:前向时信息能绕过子层(更深、更稳),反向时梯度能短路(不消失)。LayerNorm 又保证了每层激活尺度,两者一起让 Transformer 可以堆到几十上百层。

## 🎯 面试会怎么考

- **八股题**:
  - Attention 的原理?Q、K、V 各代表什么?
  - 为什么除以 $\sqrt{d_k}$?不除会怎样?
  - Multi-Head Attention 的作用?为什么要多头?
  - 为什么需要位置编码?
- **手撕题**:用 numpy 或 PyTorch 手写一个 Scaled Dot-Product Attention(含 mask 参数)。
- **深挖题**:
  - RoPE 比原始 sin/cos 编码好在哪?(相对位置 + 长度外推)
  - 为什么残差连接和 LayerNorm 缺一不可?
  - Pre-Norm 和 Post-Norm 有什么区别?为什么现在大模型普遍用 Pre-Norm?(稳定性、可训更深)
  - BERT 用 Encoder、GPT 用 Decoder,生成式大模型为什么选 Decoder-only?

## 📂 简历可写的项目

**项目名:从零实现 Multi-Head Attention 并可视化注意力权重**

- **描述**:不依赖 `torch.nn.MultiheadAttention`,用 numpy / PyTorch 手动实现 Scaled Dot-Product Attention 和 Multi-Head Attention,完整支持 Causal Mask;在一个预训练小模型(如 GPT-2 small)上 hook 出某一层的注意力矩阵,用 matplotlib 把它画成热力图,展示不同 head 学到的不同模式(语法、共指、位置等)。
- **技术栈**:PyTorch、Matplotlib、HuggingFace Transformers。
- **加分点**:对比"单头 vs 多头"在 TinyShakespeare 数据集上的训练 loss;实现 RoPE 并对比 Sinusoidal 的长度外推效果(把测试长度拉到训练长度的 2 倍)。

## 🚀 挑战

在单头 `scaled_dot_product_attention` 的基础上,实现 `multi_head_attention(Q, K, V, num_heads, mask=None)`:
1. 把 `Q/K/V` 从 `(N, d_model)` reshape 成 `(num_heads, N, d_model/num_heads)`。
2. 对每个头独立调用 `scaled_dot_product_attention`。
3. 把结果 concat 回 `(N, d_model)`,再乘一个输出投影 $W^O$。

做完后,把 `num_heads=1`、`num_heads=8`、`num_heads=64` 各跑一次,对比注意力权重热力图的差异。你会发现:头数太少 → 模式太粗;头数太多 → 每头维度太小、表达力不足。这就是为什么 LLaMA-7B 选 32 头、$d_{model}=4096$、每头 128 维——是个工程权衡。

## 🔗 延伸阅读

1. **[The Annotated Transformer(Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)** — 把原论文逐行代码化(PyTorch),边读论文边跑代码的最佳搭档。强烈建议先做一遍这个再开始任何 Transformer 项目。
2. **[The Illustrated Transformer(Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)** — 用大量动图把 Q/K/V、Multi-Head、Encoder/Decoder 全部可视化,直觉建立首选,看一遍就不会忘。
3. **[RoPE 原论文(RoFormer)](https://arxiv.org/abs/2104.09864)** — 旋转位置编码的来源,搞大模型必读。读不懂推导没关系,关键是理解"为什么相对位置 + 长度外推好"这两点。

---

## ✅ 课后小测(答案)

**课前小测答案:**
1. **为了防止 softmax 饱和**。点积 $Q\cdot K$ 的方差随 $d_k$ 线性增长,$d_k$ 大时数值会很大,softmax 进入饱和区(最大 logit 接近 1、其余接近 0),梯度几乎为 0,训练停滞。除以 $\sqrt{d_k}$ 把方差拉回 1,softmax 留在工作区。不除的话,模型在 $d_k$ 较大时根本训不动。
2. **为了学到多种不同的"关注模式"**。单头只能学一种关系(例如主谓一致),多头把向量切到多个子空间,每个子空间独立做 attention,等价于一组并行的、关注不同关系(语法、共指、位置、语义相似……)的"特征检测器"。总参数量与单头基本相同,是"免费的多样性"。
3. **Causal Mask 是上三角(对角线及以上)屏蔽**,屏蔽的是**未来 token**。把上三角置 $-\infty$,softmax 后这些位置权重为 0,保证第 $t$ 个位置只能看自己和之前的 token,从而支持自回归生成。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——

> 请默写出 Scaled Dot-Product Attention 的公式,逐项解释含义;然后说明:如果不除 $\sqrt{d_k}$、不做 Multi-Head、不加位置编码、去掉残差或 LayerNorm,模型分别会在**数值稳定性、表示能力、顺序敏感性、可训深度**上的哪一项崩掉?
>
> **答**:
> 公式 $\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$。
> - $QK^T$:Query 与 Key 的相似度矩阵。
> - $\sqrt{d_k}$:方差缩放,防 softmax 饱和。
> - $\text{softmax}$:把相似度归一化为注意力权重。
> - $\times V$:用权重加权聚合 Value,得到最终输出。
> - 不除 $\sqrt{d_k}$ → **数值稳定性**(softmax 饱和、梯度消失)。
> - 不做 Multi-Head → **表示能力**(只能学一种关注模式)。
> - 不加位置编码 → **顺序敏感性**(Attention 是集合操作,分不出语序)。
> - 去掉残差 → **可训深度**(梯度消失、堆不深)。
> - 去掉 LayerNorm → **可训深度 + 数值稳定性**(激活尺度爆炸/塌缩)。
