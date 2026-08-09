# 03 · Scaling Laws 与数据配比

> 一句话:**学完这节,你能回答「7B 模型该用多少 token 训练」和「为什么 Chinchilla 推翻了 Kaplan」这两个高频面试题。**

## 🤔 课前小测

先别急着学,花 30 秒回答下面 3 题(答案在文末):

1. GPT-3(175B 参数)用了约 300B token 训练,参数:token 大约是 1:1.7。按 Chinchilla 的结论,这个比例是偏大、偏小,还是刚好?
2. 假设你有 2 倍算力,想把模型参数翻倍(从 7B 到 14B)。按 Chinchilla,训练数据(token 数)应该翻几倍?
3. 「涌现能力」(emergent abilities)是指什么?是所有模型到了某个规模都会出现的稳定现象吗?

---

## 这节解决什么问题

Scaling Laws 是 LLM 时代最重要的经验规律之一:它告诉你**给定预算,模型该多大、数据该多少,才能把 loss 压到最低**。这听起来像工程问题,但它直接决定了几十亿美金训练预算怎么花——OpenAI 和 DeepMind 给出过两个**不同**的答案,业界为此争论了两年。

本节串起 Kaplan(2020)与 Chinchilla(2022)的分歧、Compute-Optimal 训练点、数据配比与清洗、涌现能力四件事。理解了这节,你就能看懂任何 LLM 训练论文里的「我们训了 N 个 token」是不是花得值。

## 核心概念 ★

### Kaplan Scaling Laws(OpenAI,2020)

Kaplan 等人发现:Transformer 的 test loss 随**参数量 $N$、数据量 $D$、算力 $C$** 呈**幂律**下降,跨 7 个数量级依然成立:

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_N \approx 0.076
$$

其中 $N_c, D_c$ 是常数。关键结论:**优先把算力投到更大的模型上**——Kaplan 拟合出 $N \propto C^{0.73}$,$D \propto C^{0.27}$,即算力增长时,参数增长应远快于数据。GPT-3 就是这个理论的产物:175B 参数但只用 300B token(参数:token ≈ 1:1.7)。

### Chinchilla(DeepMind,2022):纠正 Kaplan

DeepMind 训了 400+ 个模型(从 70M 到 16B,在 5B 到 400B token 上),重新拟合得到**截然不同**的结论:

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

即参数和数据应该**等比例**增长,最优配比为:

$$
\boxed{\;D_{\text{opt}} \approx 20 \cdot N_{\text{opt}}\;}
$$

**约 20 个 token 配 1 个参数**。GPT-3 的 1:1.7 严重「欠训练」——同样算力本该训小一点的模型(70B)+ 更多数据(1.4T token),效果会更好。DeepMind 用这个配方训了 Chinchilla(70B + 1.4T token),在几乎所有 benchmark 上**打败了 2.5 倍参数的 GPT-3**。

这就是 **Compute-Optimal 训练点**:在固定算力 $C$ 下,($N_{\text{opt}}, D_{\text{opt}}$)使最终 loss 最小。偏离这个点(模型太大、数据太少)就是浪费算力。

> ✅ **思考一下**:既然 Chinchilla 说 20:1,为什么 LLaMA-2 7B 用了 2T token(比例约 286:1),远超 20:1?(提示:Chinchilla 是「Compute-Optimal」,追求**单次训练**性价比;LLaMA 追求的是「给定参数量下最强模型」,愿意多花算力做 inference-friendly 的小模型。这就是 Chinchilla-optimal 和 over-train 的区别。)

### 数据配比:不同领域的影响

 Scaling Law 解决了「总量」,配比解决「结构」。同样 1T token,中英比例、代码比例、数学比例怎么分,直接决定模型擅长什么。典型配比参考(以 LLaMA / GPT-3 公开信息为近似):

| 数据类型 | 典型占比 | 对能力的影响 |
|----------|----------|--------------|
| 英文网页 | 50–70% | 通用语言能力 |
| 中文 | 5–15% | 中文理解/生成(中文 LLM 会提到 30%+) |
| 代码(GitHub) | 5–10% | **显著提升推理和数学**(代码是结构化思维) |
| 书籍/论文 | 5–10% | 长依赖、知识密度 |
| 数学 | 2–5% | 数值推理、符号操作 |
| 对话(论坛) | 3–8% | 对话风格、指令跟随 |

经验:加 5–10% 代码,推理 benchmark 涨 5–10 分;加数学语料对 GSM8K 类任务立竿见影。配比是小模型的「隐形手」,即使参数相同,不同配比训出的模型擅长领域天差地别。

### 训练数据清洗流程

「数据是新的代码」——预训练数据质量直接决定模型上限。主流流程:

1. **去重**(dedup):MinHash / SimHash 去掉重复文档和近似重复。重复数据会让模型「背」而非「学」,泛化崩坏。CommonCrawl 去重后常剩 30–50%。
2. **质量过滤**:用小模型(如 fasttext)或启发式规则过滤乱码、广告、SEO 垃圾、过短文档。
3. **毒性/安全过滤**:用分类器过滤色情、暴力、仇恨言论。
4. **PII 去识别**:去掉身份证、邮箱、电话等个人隐私。
5. **语言识别**:语言分类器(langdetect)分语种,避免小语种混入主语种语料。

高质量数据集(如 GPT-3 的 CommonCrawl + WebText2 + Books)经过重重清洗后,有效 token 数往往只有原始抓取量的 5–20%。**质量比数量重要**——这已成为共识。

### 涌现能力(Emergent Abilities)

Wei et al.(2022)观察到:某些能力(多步推理、指令跟随、in-context learning)在小模型上**几乎不存在**,在模型规模跨过某个阈值后**突然出现**——这就是「涌现」。

| 对比 | 涌现能力 | 平滑提升 |
|------|----------|----------|
| **表现** | 阈值后陡升(阶跃) | 随规模线性/幂律改善 |
| **任务例子** | 多步算术、符号操作、in-context learning | 翻译质量、语法正确率、loss |
| **指标特性** | exact match / 准确率(非线性) | 交叉熵 loss(平滑) |
| **是否可预测** | 难(阈值难提前估) | 可(幂律外推) |

但「涌现」也有争议:Schaeffer et al.(2023)指出,很多「涌现」是**评估指标**造成的假象——用 exact match 这种「全或无」指标,loss 平滑下降时也会表现为「突然会了」;换成连续指标(如 token-level 概率),能力其实是平滑增长的。所以「涌现是真是假」是当下前沿争论,面试答的时候要两面都说。

## 为什么这样设计

### 为什么 Kaplan 和 Chinchilla 结论不同

两个研究方法差异导致结论分歧:

| 维度 | Kaplan(2020) | Chinchilla(2022) |
|------|---------------|---------------------|
| **学习率调度** | 固定 LR,**不衰减到 0** | LR **余弦衰减到 0**(配合训练终点) |
| **实验范围** | 小模型外推大模型 | 400+ 模型,直接覆盖目标规模 |
| **拟合方式** | 分别拟合 $L(N), L(D)$ | 联合拟合 $L(N, D)$ |
| **结论** | $N \propto C^{0.73}$(优先大模型) | $N \propto C^{0.5}$(等比增长) |
| **数据配比** | 参数:token ≈ 1:1.7 | **参数:token ≈ 1:20** |
| **被验证情况** | GPT-3 践行,但被 Chinchilla 推翻 | Chinchilla 70B 实测打败 GPT-3 |
| **业界影响** | 推动了「大力出奇迹」 | 推动了「数据比参数重要」 |

最关键的差别是**学习率调度**:Kaplan 在固定预算下没把 LR 衰减完,导致大模型「显得」更划算(因为还没训到最优);Chinchilla 让每个模型都在自己的预算下训到最优,才得到真实的 $N$-$D$ 平衡。

### 为什么 Chinchilla 之后大家开始 over-train

Compute-Optimal 是「单次预训练性价比最高」,但**生产部署**考虑不同:

- 推理成本远大于训练成本(模型训一次,服务 billions 次请求)。
- 小模型推理便宜,所以**值得花更多算力把小模型训强**——即使偏离 Compute-Optimal 点。
- LLaMA-2 7B 用 2T token(286:1)、Mistral 7B 用 ~8T token,都是这个逻辑:**牺牲训练效率换推理效率**。

所以面试答「该用多少 token」要分清场景:**研究/Compute-Optimal → 20:1;生产/部署小模型 → 远超 20:1(50–300:1)**。

### 为什么数据清洗这么重要

Chinchilla 之后,大家都意识到:**在 20:1 配比下,数据质量就是模型质量的天花板**。CommonCrawl 原始数据 60% 是垃圾,不清洗就等于「用昂贵算力学噪声」。所以 RefinedWeb、RedPajama、SlimPajama、DCLM 这类高质量数据集成了核心竞争力,数据 pipeline 本身成了论文级别的贡献(如 Falcon 的 RefinedWeb)。

## 代码:最小实现

下面用 Python 画一条 Scaling Law 曲线(log-log 图,展示 loss 随算力下降):

```python
import numpy as np
import matplotlib.pyplot as plt

# Kaplan 式幂律: L(C) = A + B * C^(-alpha)
C = np.logspace(18, 23, 50)           # 算力 FLOPs, 从 1e18 到 1e23
A, B, alpha = 1.7, 5.0e9, 0.05        # 经验常数(示意)
L = A + B * C**(-alpha)

# 同时画 Chinchilla 的更优配比(同样算力 loss 更低)
alpha_chi = 0.05
L_chi = A * 0.85 + (B * 0.85) * C**(-alpha_chi)   # Chinchilla 系数更优

plt.figure(figsize=(7, 4.5))
plt.loglog(C, L, 'o-', label='Kaplan (2020)', markersize=3)
plt.loglog(C, L_chi, 's-', label='Chinchilla (2022)', markersize=3)
plt.xlabel('Compute C (FLOPs)')
plt.ylabel('Test Loss')
plt.title('Scaling Laws: Loss vs Compute')
plt.grid(which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('scaling_law.png', dpi=120)
plt.show()
print("曲线已保存为 scaling_law.png")
```

你会看到两条都是平滑下降的幂律曲线,Chinchilla 在同样算力下 loss 更低(因为数据配比更优)。这就是「数据配方调整」带来的「免费」提升——不增加算力,只调 $N$:$D$ 比例。

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/) 在线试跑,改 `alpha` 看 loss 下降快慢的变化。

## ⚠️ 易错点 / 面试陷阱

> ⚠️ **Chinchilla 的最优配比是「约 20 个 token : 1 个参数」,不是 1:1,也不是 1:20**。常见记错方向。GPT-3 的 1:1.7 是**欠训练**,不是「大力出奇迹的极致」。

> ⚠️ **Compute-Optimal ≠ 部署最优**。Chinchilla 给的是「单次训练性价比最高」的点。生产中为了小模型推理便宜,会 over-train(如 LLaMA-2 7B 用 2T token),这叫「chinchilla-optimal vs over-trained」的取舍。别一上来就说「7B 应该用 140B token」,那是 Compute-Optimal,不是当前主流做法。

> ⚠️ **数据质量比数量重要**。Chinchilla 之后大家发现,1T 高质量 token > 5T 噪声 token。所以「我们训了 10T token」不一定是优势,关键是有效 token 数和清洗质量。Falcon-180B 用了 3.5T 但配比/质量一般,效果被质量更优的更小模型追平。

> ⚠️ **涌现能力有争议**。「涌现」原论文(Wei 2022)观察的是 exact match 这类**非线性指标**;后续工作(Schaeffer 2023)指出换成连续指标后能力是平滑增长的。所以答面试别只说「涌现是真实的」,要补「指标选择会人为制造涌现假象,这是当前争论焦点」。

## 🎯 面试会怎么考

- **八股题**:Chinchilla 的最优 $N$:$D$ 配比是多少?Scaling Law 说了什么(三件事:loss 随 $N/D/C$ 幂律下降)?Kaplan 和 Chinchilla 为什么结论不同?
- **手撕题(画图)**:给定 $L(C) = A + B C^{-\alpha}$,画 log-log 图;给定算力预算反推 $N_{\text{opt}}, D_{\text{opt}}$。
- **深挖题**:为什么 LLaMA 选择 over-train(偏离 Chinchilla 点)?涌现能力是真的吗(Schaeffer 的反驳)?数据配比里加代码为什么能提升推理?数据去重为什么这么重要?

## 📂 简历可写的项目

**项目名:小型 Scaling Law 复现实验**

描述:在 WikiText-103 子集上训练 5 个不同规模的 Decoder-only 模型(10M / 30M / 60M / 100M / 150M 参数),每个模型在多个 checkpoint(1B / 2B / 4B token)记录验证集 loss。拟合 Kaplan 式幂律 $L(N)$、$L(D)$,画出 loss 随参数/数据的双对数曲线,验证幂律关系并估计算力-最优配比。对比「参数翻倍 + 数据按 20:1 配比」vs「只翻参数」的效果差异。

技术栈:PyTorch、nanoGPT、wandb/tensorboard、scipy(curve_fit)、matplotlib。

亮点词:**Scaling Law 复现、幂律拟合、Compute-Optimal 验证、参数-数据配比实验**。

## 🚀 挑战

**算一道真题:7B 模型按 Chinchilla 应该用多少 token 训练?如果改成 LLaMA-2 的做法(over-train),会用到多少?**

按 Chinchilla:$D \approx 20 \times N = 20 \times 7\text{B} = 140\text{B}$ token。

但 LLaMA-2 7B 实际用了 **2T token**(约 286:1),是 Chinchilla 点的 **14 倍**。算笔账:多花了 14 倍训练算力,换来的是部署时 7B 模型推理便宜得多(相比训一个 100B 的 Chinchilla-optimal 模型)。这个权衡在亿级 API 调用下回本极快。

进阶:估算一个 **70B 模型按 Chinchilla 该用多少 token**?(答案:$20 \times 70\text{B} = 1.4\text{T}$ token——这正好是 Chinchilla 论文里 70B 模型的训练量,所以 Chinchilla 70B 是名副其实 Compute-Optimal。)

## 🔗 延伸阅读

1. Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models* —— Chinchilla 原论文,必读,数据表很清晰。
2. Kaplan et al., 2020, *Scaling Laws for Neural Language Models* —— OpenAI 原版 Scaling Laws,看幂律拟合方法。
3. Wei et al., 2022, *Emergent Abilities of Large Language Models* + Schaeffer et al., 2023, *Are Emergent Abilities a Mirage?* —— 两篇对照读,理解涌现的争论。

---

## ✅ 课后小测(答案)

**课前小测答案:**

1. **1:1.7 严重偏离 Chinchilla 的最优配比(20:1),是「欠训练」**。GPT-3 是 Kaplan Scaling Law 的产物(Kaplan 主张优先堆参数,数据增长慢),Chinchilla(2022)用 400+ 模型实验证明 Kaplan 的拟合有偏差(主要因学习率没衰减到底),真实最优配比是参数:token ≈ 1:20。所以按现代认知,GPT-3 用同样算力应该训小一点的模型 + 更多数据,效果会更好——这正是 Chinchilla 70B 实测打败 GPT-3 175B 的原因。

2. **按 Chinchilla,参数和数据应等比例增长($N \propto C^{0.5}$, $D \propto C^{0.5}$)**。所以参数翻倍(×2),数据也**翻倍(×2)**,正好对应算力翻 4 倍($C = 6ND$,N 和 D 各 ×2 则 C ×4)。这正是 Chinchilla 的核心洞察:不要光堆参数,数据要同步跟上。Kaplan 的结论不同(参数 ×1.7,数据 ×1.3),但已被 Chinchilla 推翻。

3. **涌现能力是指某些任务能力(多步推理、in-context learning、指令跟随)在小模型上几乎为零,在跨过某个规模阈值后突然出现(陡升)**。但**它不是稳定的物理规律**——Schaeffer et al.(2023)指出,很多「涌现」是评估指标(exact match 这种全或无指标)造成的假象:loss 平滑下降时,用连续指标看,能力其实是平滑增长的。所以「涌现」是当前争论焦点,答时要两面都讲。

**掌握自检:** 你有 1000 PFLOP-day 的算力预算,想训一个 Decoder-only 模型。按 Chinchilla,你该选多大的模型($N$)、训多少 token($D$)?(提示:$C = 6ND$,且 $D = 20N$,联立解出 $N$。验证你的答案和 LLaMA 这类开源模型的实际选择是否一致,并解释为什么实际选择会偏离 Chinchilla 点。)
