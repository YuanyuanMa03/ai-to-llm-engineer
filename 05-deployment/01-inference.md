# 01 · 推理优化:KV Cache 与量化

> 一句话:**学完这节,你能回答「KV Cache 缓存的是什么、为什么能省计算?给一个 13B 模型算它在 2048 上下文下的 KV Cache 显存」这个面试题。**

## 🤔 课前小测
先别急着学,花 30 秒回答下面 3 题(答案在文末):
1. 自回归生成(每生成一个 token 重新跑一遍全序列)为什么「越生成越慢」?瓶颈在计算还是在显存?
2. KV Cache 缓存的到底是 Q、K、V 中的哪几个?为什么不缓存 Q?
3. 一个 FP16 的 13B 模型压到 INT4,显存大概能压到原来的多少?精度损失通常在什么量级?

---

## 这节解决什么问题

训练花几百万美元把模型训出来,只是开始。**真正落地服务时,推理(especially 自回归生成)有两个绕不开的瓶颈**:一是慢——每生成一个 token 都要把注意力从头算一遍;二是大——几十亿参数的模型光加载就吃满显存,服务成本爆炸。这两点直接决定一个公司能不能把模型跑起来、能不能扛住并发。

工程师的应对手段其实就两类:**缓存历史计算**(KV Cache)避免重复劳动,**降低权重精度**(量化 INT8/INT4)压缩存储和带宽。再叠上 vLLM 这类推理框架的 PagedAttention、Continuous Batching,把吞吐再拔一个数量级。这是「大模型应用 / 推理工程师」岗每天都要打的仗,也是面试高频区。

## 核心概念 ★

### 1. 自回归生成的瓶颈

Decoder-only 模型生成时,**每来一个新 token,前向都要重新计算注意力**。注意力的核心是对历史所有 token 算 $QK^T$:

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

如果每次生成都重算所有历史 token 的 K 和 V,代价随序列长度**平方增长**。生成 1000 个 token 时,前面那 999 个 token 的 K、V 被重算了 999 次——纯浪费。

### 2. KV Cache:把历史的 K、V 存起来

观察:在第 $t$ 步生成时,**历史 token 的 K、V 与新 token 无关,不会变**。所以第一次算出来后直接缓存,后面每步只算**新 token 自己的 Q** 去查这张缓存表。

$$
\underbrace{K_{1:t}, V_{1:t}}_{\text{缓存里已有的历史}} \;\oplus\; \underbrace{k_t, v_t}_{\text{新 token 的 K/V}} \;\Rightarrow\; \text{新的 }K_{1:t}, V_{1:t}
$$

省下的计算:**不再重算历史**。第 $t$ 步只需算一个 $1\times d$ 的 $q_t$ 和一个 $t\times d$ 的 $K_{1:t}$ 的注意力,复杂度从 $O(t^2)$ 降到 $O(t)$。

> ✅ **思考一下**:既然缓存的是 K、V,为什么不缓存 Q?提示:第 $t$ 步只有新 token 需要去"查询"历史,**历史的 Q 再也不会被任何人用到了**——它只是历史在"被查",不需要主动去"查别人"。

### 3. KV Cache 的显存代价

天下没有免费午餐。缓存省了计算,代价是吃显存。一个注意力头每存一个 token 要存 K 和 V 两个向量,每个 $d$ 维。多Transformer 层、多头、批处理下:

$$
\text{KV Cache 显存}=2\times L \times n_{kv}\times d \times \text{seq\_len}\times \text{batch}\times \text{dtype\_size}
$$

- `2`:K 和 V 两个张量
- $L$:层数(`num_hidden_layers`)
- $n_{kv}$:KV 头数(GQA/MQA 下小于 query 头数)
- $d$:每个头的维度(`head_dim`)
- `seq_len`:序列长度(含 prompt + 已生成)
- `batch`:并发请求数
- `dtype_size`:每参数字节数(FP16=2,FP32=4)

记忆口诀:**「2 倍 × 层 × KV 头数 × 头维 × 长度 × batch × dtype」**。

### 4. PagedAttention(vLLM)

KV Cache 在真实服务里有个工程噩梦:**请求长度不一 → 显存碎片**。短请求腾出来的洞,长请求塞不进去。

vLLM 借鉴操作系统的**虚拟内存分页**,把 KV Cache 切成固定大小的 **block**(每 block 存固定数量 token 的 K/V),按需分配、逻辑连续物理离散。一个请求的 KV 用一张 block table 映射,大大减少碎片,让 batch 能塞更多请求。这是 vLLM 比 HuggingFace 原生快数倍的核心原因之一。

### 5. Continuous Batching(动态批处理)

传统批处理是"凑齐一批一起进、一起出",慢请求拖死整批。Continuous Batching **在每一步把已结束的请求踢出、把新请求塞进来**,**token 级动态拼 batch**。配合 PagedAttention,vLLM 的吞吐能比 `transformers` 原生推理高 10x+。

### 6. 量化:把 FP16 权重压到 INT8/INT4

权重存的是浮点数(FP16/BF16,每参数 2 字节)。量化就是把它**线性映射**到低精度整数:

$$
q=\text{round}\!\left(\frac{x}{\text{scale}}\right),\qquad x_{\text{dequant}}=q\times \text{scale}
$$

`scale` 通常取 $\frac{\max(|x|)}{2^{b-1}-1}$(对称量化),把权重范围挤进 INT8(±127)或 INT4(±7)。

- **INT8**:2 字节 → 1 字节,显存减半,精度几乎不掉(<1%)。
- **INT4**:2 字节 → 0.5 字节,显存压到 1/4,精度掉 1~3 个点(任务相关)。
- **激活也可以量化**,但训练后激活分布有 outlier,处理更复杂( SmoothQuant、LLM.int8() 都是为此)。

### 7. AWQ vs GPTQ:两种训练后量化方法

| 方法 | 思路 | 特点 |
|---|---|---|
| **GPTQ** | 基于**二阶信息**(Hessian)逐列最小化量化误差,需要少量校准数据 | 精度好,但量化过程稍慢 |
| **AWQ** | 发现**保护"显著权重"(salient channels)** 比全部同等量化更重要,通过 per-channel scaling 实现 | 速度快、显存友好,推理常用 |
| **RTN**(Round-to-Nearest) | 最朴素,直接四舍五入 | INT8 还行,INT4 掉点严重 |

> ✅ **思考一下**:同样 4 比特,为什么 AWQ/GPTQ 比 RTN 掉点少很多?提示:它们都在**最小化"量化前后输出的差异"**,而不是只让权重本身尽量接近——前者关心的是**对下游的影响**,后者只看局部。

---

## 为什么这样设计

### 有 KV Cache vs 无 KV Cache

| 维度 | 无 KV Cache(朴素重算) | 有 KV Cache |
|---|---|---|
| 第 $t$ 步计算量 | $O(t^2)$(重算全部历史注意力) | $O(t)$(只算新 token 对历史一次查询) |
| 显存 | 只存模型权重 | 额外存 KV Cache,随 `seq_len × batch` 线性增长 |
| 适合阶段 | **Prefill**(首 token,无法避免) | **Decode**(逐 token 生成,纯赚) |
| 长序列风险 | 计算爆炸 | **显存爆炸**(KV Cache 比 13B 权重还大很常见) |

### FP16 vs INT8 vs INT4

| 精度 | 每参数字节 | 13B 模型显存 | 速度 | 典型精度损失 |
|---|---|---|---|---|
| FP16(基线) | 2 | ~26 GB | 1× | 0 |
| INT8 | 1 | ~13 GB | ~1.5-2× | <1% |
| INT4 | 0.5 | ~7 GB | ~2-4× | 1-3% |
| INT3 | 0.375 | ~5 GB | 更快 | 5%+,明显掉点 |

可以看到:**INT8 几乎白嫖;INT4 是性价比拐点;再往下到 INT3,精度坍塌**。这也是为什么开源社区主推 INT4 量化的本地部署。

### Prefill vs Decode:两个阶段,瓶颈不同

| 阶段 | 干什么 | 计算密集 or 访存密集 |
|---|---|---|
| **Prefill** | 一次性处理 prompt,算出第一个 token 和初始 KV Cache | **计算密集**(并行算很多 token) |
| **Decode** | 每步生成 1 个 token,要读全部权重 + KV Cache | **访存密集**(算得少但读得多) |

这个区分极重要:**量化主要省 Decode 的访存**——读 0.5 字节比 2 字节快 4 倍,所以 INT4 推理快。也解释了为什么 Continuous Batching 能提吞吐:Decode 阶段每个请求只算 1 个 token,GPU 算力闲置,正好塞更多并发。

### Prefill 和 Decode 还可以用不同 batch 策略

进一步拆开看,两个阶段的资源画像几乎相反,所以现代推理引擎(vLLM、TensorRT-LLM)对它们用不同打法:

| 策略 | Prefill 怎么打 | Decode 怎么打 |
|---|---|---|
| batch | 小(算力瓶颈,batch 大了显存爆) | 大(算力闲置,batch 越大吞吐越高) |
| 切分 | **Chunked Prefill**:把长 prompt 切成段,避免单请求占满 GPU | 不需要切,本身就只算 1 个 token |
| 调度 | 优先级低(可排队) | 优先级高(用户在等输出) |

**Chunked Prefill** 是 vLLM 后来加的关键优化:一个 4K 长的 prompt 不再一次性吃掉整张卡的算力,而是切若干 chunk 算,期间还能插别的请求的 Decode,整体吞吐再提一截。面试里能讲到这里,基本说明你读过 vLLM 的 release notes。

### 量化的真实工程权衡

精度表里"INT4 掉 1-3 个点"是**平均**。落到不同任务,差异巨大:

| 任务类型 | INT4 掉点幅度 | 原因 |
|---|---|---|
| 闲聊、常识问答 | 几乎无感(<1%) | 模型本来就不需要精确数值 |
| 文本摘要、翻译 | 轻微(1-2%) | 主要靠语言能力,权重轻微扰动影响小 |
| 数学(GSM8K)、代码(HumanEval) | 明显(3-8%) | 依赖少数关键权重的高精度 |
| 长上下文检索 | 可观 | KV Cache 量化叠加,远处 token 信息被进一步压 |

所以工程上常见的组合是:**权重 INT4 + 激活 FP16(W4A16)+ KV Cache FP16**,精度损失最小、速度收益最大;只有显存真的吃紧时才进一步量化 KV Cache(KV INT8 / INT4),且要评测掉点能否接受。

## 代码:最小实现

下面用 Python 模拟 KV Cache 的核心逻辑。**不依赖任何框架,只演示"不重算历史"这件事**,10 行就能看清本质。

```python
import numpy as np

# 模拟一个 token 的 K/V 计算函数(真实里是 W_k @ x / W_v @ x)
def compute_kv(token_emb, W_k, W_v):
    k = W_k @ token_emb   # shape: (d,)
    v = W_v @ token_emb   # shape: (d,)
    return k, v

# === KV Cache 的核心:增量更新,不重算历史 ===
class KVCache:
    def __init__(self):
        self.keys = []     # 存历史 K
        self.values = []   # 存历史 V

    def update(self, new_k, new_v):
        # 新 token 的 K/V 直接 append,历史的 K/V 一字不动
        self.keys.append(new_k)
        self.values.append(new_v)

    def attention(self, q):
        # 只有新 q 去查缓存里的所有历史 K/V
        K = np.stack(self.keys)              # (t, d)
        V = np.stack(self.values)            # (t, d)
        scores = K @ q / np.sqrt(len(q))     # (t,)  ← 只算一次点积
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        return weights @ V                   # (d,)
```

跑一个对比:**朴素重算 vs KV Cache**,看计算量差距。

```python
d = 64
seq_len = 100
W_k = np.random.randn(d, d)
W_v = np.random.randn(d, d)
tokens = [np.random.randn(d) for _ in range(seq_len)]

# --- 朴素重算:每生成一个新 token 都重算全部历史 ---
def naive_attention(tokens, W_k, W_v, new_q):
    K = np.stack([W_k @ t for t in tokens])   # 重算所有历史 K!
    V = np.stack([W_v @ t for t in tokens])   # 重算所有历史 V!
    scores = K @ new_q / np.sqrt(d)
    w = np.exp(scores - scores.max()); w /= w.sum()
    return w @ V

# --- KV Cache:历史已缓存,只算新 token ---
cache = KVCache()
for t in tokens:
    cache.update(*compute_kv(t, W_k, W_v))

new_q = np.random.randn(d)
out_naive = naive_attention(tokens, W_k, W_v, new_q)
out_cache = cache.attention(new_q)
print("结果一致:", np.allclose(out_naive, out_cache, atol=1e-6))  # True
# 但 naive 在 seq_len=100 时算了 100 次投影,KVCache 只算 1 次
```

再补一个**显存估算函数**,面试手撕常考。

```python
def kv_cache_bytes(num_layers, num_kv_heads, head_dim,
                   seq_len, batch, dtype_bytes=2):
    """KV Cache 显存公式:2 * L * n_kv * d * seq_len * batch * dtype"""
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * batch * dtype_bytes

# Llama-2-13B:L=40, n_kv=40(GQA 后实际看 config), d=128, batch=8, seq=2048
# 这里按一般配置演示,以官方 config.json 为准
mem = kv_cache_bytes(num_layers=40, num_kv_heads=40, head_dim=128,
                     seq_len=2048, batch=8, dtype_bytes=2)
print(f"KV Cache: {mem/1024**3:.2f} GB")  # 看 batch=8 时多大
```

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/) 在线试跑,把 `seq_len` 从 512 调到 8192,看显存怎么爆。

## ⚠️ 易错点 / 面试陷阱

> **KV Cache 是「加速 Decode」的,不是「省 Prefill」的。** Prefill 阶段所有 token 都要算,缓存还没建立,无缓存可省。所以"用了 KV Cache 首字延迟就快了"是错的——首字延迟主要被 Prefill 计算 + 权重加载决定。

> **KV Cache 对长序列显存爆炸,常常比模型权重还大。** 很多人直觉觉得"模型 13B,显存顶天就 26GB",其实 batch=32、seq=4096 时 KV Cache 可能超过权重本身。这也是为什么有 GQA / MQA——它们**减少 KV 头数**直接砍 KV Cache,代价是轻微掉点。

> **INT4 不是 4 个整数,而是 4-bit。** 一个字节 8 位可以塞 2 个 INT4 权重。算显存时是 0.5 字节/参数,不是 1 字节。算错这一步,模型能不能塞进单卡都判错。

> **量化掉点不是均匀的,数学/代码任务更敏感。** INT4 在常识问答、对话上几乎无感,但在需要精确计算的数学/代码任务上掉点明显。所以"INT4 掉 2 个点"是平均,具体看你评测集。

> **激活量化比权重量化难。** 权重分布稳定(训完就固定),激活里有 outlier(LLM.int8() 论文发现少数通道数值远超均值),朴素 INT8 激活量化会塌。所以工程上常常**权重 INT4 + 激活 FP16**(W4A16),就是 AWQ / GPTQ 实际部署的形态。

> **GQA / MQA 不是免费午餐,它在精度上付代价。** 它们通过减少 KV 头数砍 KV Cache,但 KV 头变少意味着多个 query 头共享同一组 K/V,表达能力下降。GQA(分组,如 8 组)是质量和显存的折中,MQA(全部 query 头共享 1 组 K/V)最省但掉点最多。Llama-2-70B、Qwen-2 都用 GQA 而非 MQA,就是这个权衡。面试被问"为什么不全用 MQA",答这个。

> **batch 不是越大越好,有吞吐拐点。** Decode 阶段 batch 越大吞吐越高——直到 KV Cache 把显存吃满,或 GPU 算力被打满。再往上加 batch,要么 OOM,要么吞吐不涨反而延迟飙升。生产部署要画「并发数 vs 吞吐 vs P99 延迟」三曲线找拐点,不是无脑拉 batch。

## 🎯 面试会怎么考

- **八股题**:KV Cache 原理?缓存的为什么是 K、V 而不是 Q?为什么 GQA / MQA 能省 KV Cache?量化为什么 INT8 几乎不掉点、INT4 才明显?
- **手撕题**:给一个模型配置($L$、$n_{kv}$、$d$、`seq_len`、`batch`),算 KV Cache 显存;或写一个 `update_cache(new_k, new_v)` 函数和 `attention(q, cache)` 函数。
- **深挖题**:vLLM 的 PagedAttention 怎么减少碎片?Continuous Batching 和静态 batching 的吞吐差距在哪?Prefill 和 Decode 哪个是访存密集,为什么量化对 Decode 加速更明显?AWQ 和 GPTQ 思路差异?

## 📂 简历可写的项目

**项目名:基于 vLLM 的开源大模型推理服务部署与性能对比**

- **描述**:用 vLLM 部署 Llama-2-13B(或 Qwen / ChatGLM),启用 PagedAttention + Continuous Batching,对比 `transformers` 原生 `generate` 的吞吐(tokens/s)和首字延迟;进一步加载 AWQ INT4 量化版本,记录显存下降和精度变化(MMLU / 自建评测)。
- **技术栈**:vLLM、HuggingFace transformers、AWQ / GPTQ、numpy、nvidia-smi / torch.cuda 监控。
- **加分点**:画出「并发请求数 vs 吞吐」曲线,指出拐点;量化前后在 5 个评测任务上做对比;写一篇部署踩坑笔记(如 `gpu_memory_utilization` 调参、长序列 OOM 处理)。

## 🚀 挑战

**算一个数**:Llama-2-13B 在 `batch=1`、`seq_len=2048` 下,KV Cache 占多少显存?(以官方 config 的 `num_hidden_layers=40`、`num_key_value_heads=40`、`head_dim=128`、FP16 为准。)

提示:
1. 先算权重本身:13B × 2 字节 = 26 GB。
2. 再套公式 `2 × 40 × 40 × 128 × 2048 × 1 × 2`。
3. 看看 KV Cache 占了权重的几分之几。如果把 `batch` 提到 32,会发生什么?

参考答案:batch=1 时约 1.56 GB,占权重 6%,单卡完全装得下;batch=32 时约 50 GB,**接近 13B 权重(26 GB)的两倍**,单张卡吃紧——这就是为什么生产 RAG / 长上下文服务必须上 GQA(砍 KV 头)、量化 KV(KV INT8)、PagedAttention(减少碎片多塞请求),或者干脆多卡张量并行。

进阶追问:如果同样的负载换到 Llama-2-70B($L=80$、$n_{kv}=8$ 的 GQA 配置),KV Cache 反而比 13B 更小,为什么?(提示:GQA 把 KV 头数从 40 砍到 8,降幅远超层数从 40 涨到 80 的增幅。)

进阶:把 batch 调到 32,KV Cache 会不会超过 26GB?如果会,你能想到哪几种应对方案(GQA、量化 KV、PagedAttention、把 KV Cache 卸到 CPU)?

## 🔗 延伸阅读

1. 📄 [Efficient Memory Management for Large Language Model Serving with PagedAttention(vLLM 论文,SOSP 2023)](https://arxiv.org/abs/2309.06180) —— PagedAttention 的原始出处,把"虚拟内存分页"思路用到 KV Cache 的开山之作。
2. 📄 [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) —— "保护显著权重"的 INT4 量化方法,工程上最常用的量化方案之一。
3. 💻 [vLLM 官方文档与 GitHub](https://github.com/vllm-project/vllm) —— 推理框架事实标准,Issues 里全是真实部署踩坑案例,看完能直接聊生产级推理。

---

## ✅ 课后小测(答案)

**课前小测答案:**
1. **自回归越生成越慢,根因是每步重算历史注意力,计算量 $O(t^2)$ 增长;但更隐蔽的瓶颈是 Decode 阶段访存密集——每算一个 token 都要把全部权重 + KV Cache 读一遍,GPU 算力吃不饱,显存带宽卡脖子。**
2. **缓存的是 K 和 V。Q 不缓存,因为历史 token 在后续步骤里只"被查询"(贡献 K),不再"查询别人"(不需要 Q),它的 Q 永远不会再被使用。**
3. **FP16 → INT4,显存压到原来的 1/4(13B 从 26GB 降到约 7GB);精度损失通常 1-3 个点(平均),数学/代码任务更敏感。**

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——

> 给定 Llama-2-13B($L=40$、$n_{kv}=40$、$d=128$、FP16),`batch=1`、`seq_len=2048`,算 KV Cache 占多少显存;然后说明:如果改成 GQA 把 $n_{kv}$ 砍到 8,KV Cache 能省多少?为什么这种省法在工程上比单纯砍 batch 更受欢迎?
>
> **答**:
> - 套公式:$2 \times 40 \times 40 \times 128 \times 2048 \times 1 \times 2 = 1{,}677{,}721{,}600$ 字节 ≈ **1.56 GB**。
> - 占 13B 权重(26 GB)的约 6%,单卡毫无压力;但 `batch=32` 时飙升到 ~50 GB,**接近权重的两倍**——这就是长上下文服务的显存炸弹。
> - GQA 把 $n_{kv}$ 砍到 8,KV Cache 变成原来的 $8/40=1/5$,约 0.31 GB,**直接 5 倍节省**。
> - 砍 batch 是牺牲吞吐(并发变少),GQA 是几乎不牺牲吞吐的"结构性省法",只在精度上付微小代价——服务并发越多越赚,这也是 Llama-2-70B / Qwen-2 都用 GQA 的原因。
>
> **一句话收尾**:KV Cache 的本质是"用显存换计算",量化的本质是"用精度换显存带宽",PagedAttention / Continuous Batching 的本质是"用工程换吞吐"——推理优化永远是这三组 trade-off 的组合拳。
