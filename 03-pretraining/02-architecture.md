# 02 · LLM 架构:Decoder-only 与 MoE

> 一句话:**学完这节,你能回答「为什么 GPT 用 Decoder-only」和「MoE 凭什么省算力」这两个架构面必考题。**

## 🤔 课前小测

先别急着学,花 30 秒回答下面 3 题(答案在文末):

1. BERT 是 Encoder-only,GPT 是 Decoder-only。为什么生成任务(写文章、对话)用 Decoder,而理解任务(分类、抽取)用 Encoder?关键区别是什么?
2. 现代主流 LLM(GPT-4、LLaMA)都用 Pre-LN 而不是 Post-LN。两者差在哪一步?为什么 Pre-LN 训练更稳?
3. MoE(混合专家)模型有 8 个专家每次只激活 2 个。它「省」的是训练算力、推理算力,还是两者都省?

---

## 这节解决什么问题

Transformer 是 LLM 的地基,但「Transformer」本身只是一个骨架。真正决定模型行为的是:用 Encoder / Decoder / 还是两者都用?位置编码怎么加?归一化放哪?注意力怎么分组?要不要上 MoE?这些选择题组合起来,就是 GPT-4 / LLaMA / Mixtral 的架构差异。

本节串起 Decoder-only 主流化、关键组件(RoPE / RMSNorm / SwiGLU / GQA)、MoE 稀疏激活这三条线,让你看到任何一份 LLM 论文都能对号入座。

## 核心概念 ★

### 三大 Transformer 架构

| 架构 | 注意力 | 代表模型 | 擅长 | 训练目标 |
|------|--------|----------|------|----------|
| **Encoder-only** | 双向(看全文) | BERT、RoBERTa | 理解:分类、NER、检索 | MLM(盖住预测) |
| **Decoder-only** | 单向(Causal Mask) | **GPT、LLaMA、Qwen** | 生成:对话、续写、代码 | **预测下一个 token** |
| **Encoder-Decoder** | 编码双向+解码单向 | T5、BART | 翻译、摘要(seq2seq) | Span corruption / 去噪 |

Decoder-only 成主流的两个原因:

1. **训练效率高**。预测下一个 token 这种自监督目标对**每个位置**都产生梯度信号(序列长 $L$ 就有 $L$ 个训练样本);MLM 只对被 mask 的 ~15% 位置有信号。同样算力下 Decoder-only 学得更充分。
2. **Scaling 表现好**。Kaplan / Chinchilla 的实验都显示,在「参数 + 数据 + 算力」同步放大时,Decoder-only 的 loss 下降最平滑,GPT-3 的涌现能力也证明了这点。

### 关键组件 1:RoPE 旋转位置编码

Transformer 原版用**绝对位置编码**(把位置 $0,1,2,...$ 加到 Embedding 上)。问题:位置 5 和 10 的关系学不到,长序列外推差。RoPE(Rotary Position Embedding,苏剑林 2021,LLaMA 系标配)用**旋转**把相对位置编进 $q, k$:

$$
RoPE(x, m) = x \cdot e^{im\theta}
$$

把 $d$ 维向量两两一组视为复数,乘以角度 $m\theta$(位置 $m$ 决定转多少)。关键性质:

$$
\langle RoPE(q, m),\, RoPE(k, n) \rangle = \text{Re}\left((q e^{im\theta})(k e^{in\theta})^*\right) = f(q, k, m-n)
$$

即 attention 的内积结果**只依赖相对位置 $m - n$**,天然支持长序列外推,且无需额外参数。

### 关键组件 2:RMSNorm(去 centering 的 LayerNorm)

LayerNorm 同时做减均值(centering)和除标准差:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta, \quad \mu = \frac{1}{n}\sum x_i, \; \sigma^2 = \frac{1}{n}\sum(x_i - \mu)^2
$$

RMSNorm(LLaMA 用)发现 centering 没必要,只保留除 RMS:

$$
\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}, \quad \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

省掉 $\mu$ 的计算和 $\beta$ 参数,速度快 ~10–20%,效果几乎无损。

### 关键组件 3:SwiGLU 激活

标准 FFN 是 $\text{FFN}(x) = \text{GELU}(xW_1)W_2$。SwiGLU(LLaMA 用)给 FFN 加一个门控(gating):

$$
\text{SwiGLU}(x, W, V) = (\text{Swish}(xW) \otimes xV), \quad \text{Swish}(x) = x \cdot \sigma(\beta x)
$$

$\otimes$ 是逐元素乘。直觉:让一部分信号通过(Swish 门),另一部分直接乘过去(V),比单一非线性表达能力更强。代价是参数多了 $1/3$(三层矩阵),所以 LLaMA 把 FFN 隐藏维从 $4d$ 调到 $\frac{2}{3} \cdot 4d$ 来平衡。

### 关键组件 4:GQA 分组注意力

MHA(Multi-Head Attention):$n_{heads}$ 个 Query 头 + $n_{heads}$ 个 KV 头,每个 Q 头配一份 K/V。问题:**KV Cache 显存 = $2 \times n_{heads} \times d_{head} \times L$**,头多就爆。

| 方案 | Q 头数 | KV 头数 | KV Cache | 代表模型 |
|------|--------|---------|----------|----------|
| **MHA** | $h$ | $h$ | $2hdL$ | GPT-2、BERT |
| **MQA** | $h$ | 1 | $2dL$ | PaLM、StarCoder |
| **GQA** | $h$ | $g$(分组) | $2gdL$ | **LLaMA-2/3、Qwen2** |

MQA 把所有 Q 头共享一份 KV,显存最小但效果略掉;GQA 折中——$h$ 个 Q 头分成 $g$ 组,每组共享一份 KV。LLaMA-2 70B 用 GQA(g=8),既省显存又不掉点。

### MoE:稀疏激活的混合专家

Dense 模型每个 token 都过所有参数。MoE(Mixture of Experts)把 FFN 换成 $N$ 个并行的「专家」FFN,每次只激活 Top-K 个:

$$
y = \sum_{i \in \text{Top-K}(\text{gate}(x))} \text{gate}_i(x) \cdot \text{Expert}_i(x)
$$

$\text{gate}(x)$ 是一个小的路由网络(通常线性层 + softmax),输出每个专家的得分,选 Top-K(常见 K=2)。Mixtral 8×7B 是 8 个专家每次激活 2 个,总参数 47B 但**激活参数只有 13B**——推理速度对标 13B Dense,质量对标 70B Dense。

> ✅ **思考一下**:MoE 的「总参数」和「激活参数」哪个决定推理算力成本?(答案:**激活参数**决定 FLOPs,所以 MoE 推理便宜;但所有专家参数都要驻显存,所以**显存成本由总参数决定**——这就是 MoE 推理的「显存-算力」剪刀差。)

## 为什么这样设计

### 为什么 Decoder-only 赢了

- **数据效率**:预测下一个 token 是天然的密集监督,互联网所有文本都能拿来训,不需要标注。
- **任务通用**:同一个 next-token 目标可以零样本做翻译、摘要、写代码;Encoder-only 想做生成得改结构。
- **Scaling 一致**:从 125M 到 175B,GPT-3 证明 loss 平滑下降,能力涌现稳定;Encoder-Decoder 在大模型上没显示出同等优势。

### Pre-LN vs Post-LN

Post-LN(原版 Transformer):`x → Attn → Add → LN → FFN → Add → LN`,LN 在残差外。问题:深了之后梯度不稳,需要 warmup 才训得动。

Pre-LN(GPT-2、LLaMA):`x → LN → Attn → Add → LN → FFN → Add`,LN 在残差内,主路是干净残差。深模型训练更稳,可省掉复杂 warmup。

| 对比 | Post-LN | Pre-LN |
|------|---------|--------|
| LN 位置 | 残差外 | 残差内 |
| 主路 | 被归一化打断 | 干净残差(恒等映射) |
| 深层训练 | 不稳,需长 warmup | **稳定** |
| 最终效果 | 略好一点点 | 实践首选(够稳) |

### Dense vs MoE

| 对比 | Dense(如 LLaMA-2 70B) | MoE(如 Mixtral 8×7B) |
|------|------------------------|------------------------|
| **总参数量** | 70B | 47B |
| **激活参数** | 70B(全激活) | 13B(每次 2/8) |
| **训练算力** | 与参数成正比 | 与**激活**成正比,省 |
| **推理 FLOPs** | 高(= 参数量) | 低(= 激活参数) |
| **推理显存** | = 参数量 | **= 总参数量(全部专家驻留)** |
| **负载均衡** | 无此问题 | **关键挑战**(专家冷热不均) |

MoE 的核心痛点是**负载均衡**:Router 容易把绝大多数 token 都路由到同一个「热门专家」,其他专家空转。解法是训练时加一个辅助 loss,惩罚专家选择的不均匀度。

### GQA vs MHA vs MQA

| 维度 | MHA | MQA | GQA |
|------|-----|-----|-----|
| KV 头数 | $=h$ | $=1$ | $=g$(中间) |
| KV Cache 显存 | 最大 | 最小 | 中间 |
| 效果 | 最好 | 略掉 | 接近 MHA |
| 长序列/批处理 | 显存爆 | 友好 | **友好** |

GQA 是性价比之王,所以 LLaMA-2/3、Qwen2 都用。

## 代码:最小实现

下面用 numpy 示意 **MoE 的 Top-2 路由**,15 行看懂稀疏激活:

```python
import numpy as np

np.random.seed(0)
d, n_experts, top_k = 16, 4, 2          # 隐藏维16, 4个专家, 激活2个
n_tokens = 6                            # 一个 batch 6 个 token

x = np.random.randn(n_tokens, d)        # 模拟上一层输出
W_gate = np.random.randn(d, n_experts)  # 路由网络
W_experts = np.random.randn(n_experts, d, d)  # 4 个专家的 FFN 权重

# 1. Router 给每个 token 算每个专家的得分
logits = x @ W_gate                     # (n_tokens, n_experts)
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

# 2. 每个 token 选 Top-K 个专家
out = np.zeros_like(x)
for t in range(n_tokens):
    top_idx = np.argsort(probs[t])[-top_k:][::-1]   # 得分最高的 K 个
    top_w = probs[t][top_idx]
    top_w = top_w / top_w.sum()                     # 归一化权重
    for idx, w in zip(top_idx, top_w):
        out[t] += w * (x[t] @ W_experts[idx])       # 加权融合专家输出

# 看每个 token 被路由到哪两个专家
for t in range(n_tokens):
    print(f"token {t}: 专家 {np.argsort(probs[t])[-top_k:][::-1]}, 权重 {np.sort(probs[t])[-top_k:][::-1]}")
```

跑一遍你会看到不同 token 选了不同专家组合——这就是稀疏激活的本质。把 `top_k` 改成 `n_experts` 就是 Dense FFN。

> 💡 点「编辑」改 `n_experts` 和 `top_k`,再「运行到这里」观察路由分布。

## ⚠️ 易错点 / 面试陷阱

> ⚠️ **Decoder-only 必须用 Causal Mask**。Self-attention 里加一个上三角 mask($-\infty$),让位置 $i$ 只能看到 $\le i$ 的位置,否则就是「作弊」(看到了未来 token)。这是 Decoder 和 Encoder 最本质的结构差异,不是「参数共享不同」。

> ⚠️ **Pre-LN 不是「把 LN 放前面就完了」**,主路变成恒等残差才是关键。这意味着深层网络输出 = 输入 + 各层贡献的累加,梯度可以无衰减地回传,所以稳。代价:深层等效于浅层堆叠,表达能力略弱,但实践上稳定性更重要。

> ⚠️ **MoE 省的是「算力」不是「显存」**。所有专家参数都得加载到显存(或对应存储层级),只是推理时只前向过被选中的那几个。所以 MoE 对显存带宽敏感,常是 memory-bound。面试常见反问:「那 MoE 还有什么贵的地方?」——答:专家路由的通信开销(多卡部署时 token 要在不同卡之间 All-to-All 交换)。

> ⚠️ **RoPE 不增加参数**。它是 $q, k$ 上的一个固定旋转(由位置 $m$ 决定),没有可学习参数。绝对位置编码才有参数表(`nn.Embedding(max_len, d)`)。

## 🎯 面试会怎么考

- **八股题**:Decoder-only 为什么成主流?讲一下 RoPE 的原理(为什么内积只依赖相对位置)。MoE 怎么省算力的?SwiGLU 比普通 FFN 强在哪?Pre-LN 为什么比 Post-LN 稳?
- **手撕题**:画出一个 Decoder-only 层的完整前向流程(含 Causal Mask、RoPE、RMSNorm、SwiGLU 的位置);用伪代码写 MoE 的 Top-K 路由;解释 GQA 怎么共享 KV。
- **深挖题**:GQA 为什么比 MHA 省 KV Cache(算一下显存)?MoE 的负载均衡 loss 怎么设计?RoPE 怎么支持长序列外推(NTK-aware / YaRN 是在改什么)?RMSNorm 去 centering 为什么不掉点?

## 📂 简历可写的项目

**项目名:nanoGPT 源码精读与 Decoder-only 数据流可视化**

描述:精读 Karpathy 的 nanoGPT(~300 行 PyTorch),逐模块梳理一个 Decoder-only 层的前向数据流:Token Embedding + 位置编码 → Causal Self-Attention(含 GQA 改造)→ RMSNorm → SwiGLU FFN → 残差。产出一张完整数据流图(标注张量 shape、参数量、KV Cache 注入点),并实现一个小改动:把 MHA 换成 GQA,在 Shakespeare 数据集上对比显存和 loss 曲线。

技术栈:PyTorch、nanoGPT、torch.profile、draw.io。

亮点词:**源码精读、架构改造(MHA→GQA)、显存 profiling、KV Cache 分析**。

## 🚀 挑战

算一道面试真题:**LLaMA-2 70B 和 LLaMA-2 7B 在 4096 序列长度、batch=1 下,KV Cache 各占多少显存?**(提示:KV Cache = $2 \times n_{kv\_heads} \times d_{head} \times L \times n_{layers} \times 2$ bytes(fp16))。

参数:
- 7B:32 层,32 头,$d_{head}=128$,MHA
- 70B:80 层,64 Q 头 + 8 KV 头(GQA),$d_{head}=128$

算完你会发现为什么 70B 上长上下文那么吃显存,也就理解了 GQA 的价值。把答案换算成 GB。

## 🔗 延伸阅读

1. 苏剑林博客《Transformer 升级之路:博采众长的 RoPE》—— RoPE 作者本人讲原理,中文最权威。
2. Ainslie et al., 2023, *GQA: Training Generalized Multi-Query Transformer Models* —— GQA 原论文,实验对比 MHA/MQA/GQA。
3. Karpathy, nanoGPT GitHub —— 300 行读懂 Decoder-only 全貌,比读大仓库高效 10 倍。

---

## ✅ 课后小测(答案)

**课前小测答案:**

1. **关键区别是注意力方向**。Encoder 用双向注意力(每个位置看到全文),适合**理解**任务——分类、NER、检索都需要上下文;Decoder 用 Causal Mask(只看左侧历史),适合**生成**任务——按从左到右的顺序预测下一个 token,不能偷看未来。BERT 的 MLM 目标(盖住中间预测)需要双向,但生成时没有「未来」可参考,所以 Encoder 不擅长自回归生成。

2. **Pre-LN 把 LN 移到残差分支内,主路保持干净残差**。Post-LN 是 `Add(LN(Attn(x)), x)`,主路被 LN 打断;Pre-LN 是 `Add(Attn(LN(x)), x)`,主路是恒等映射。Pre-LN 更稳的原因:深层网络主路 $x_{l+1} = x_l + f(x_l)$ 是恒等累加,梯度可以无衰减回传,不会随深度爆炸/消失。代价是深层等效于浅层叠加,效果略损,但训练稳定性更重要,所以现代 LLM 都用 Pre-LN。

3. **MoE 省的是「推理算力(FLOPs)」**,因为每个 token 只过 Top-K 个专家而非全部;**训练算力也省**(每个 token 的前向/反向都只算激活专家)。但**显存不省**——所有 $N$ 个专家的参数都要驻留在显存,推理时按总参数算显存,按激活参数算 FLOPs。这就是 MoE 的「显存-算力剪刀差」,也是为什么 Mixtral 8×7B 总参 47B 但推理速度像 13B。

**掌握自检:** 给定一个 4 层 Decoder-only 模型,每层有 MHA(8 头,$d_{head}=64$)+ FFN($4d$)+ LayerNorm。写出输入 `[batch=2, seq=16, d=512]` 经过一层后每个中间张量的 shape,并标注 Causal Mask 注入的位置。
