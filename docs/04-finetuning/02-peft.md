# 02 · PEFT:LoRA 与 QLoRA 的原理

> 一句话:**学完这节,你能回答「LoRA 为什么能省显存、为什么初始化时 B=0、QLoRA 比 LoRA 多了什么」这三个面试高频题。**

## 🤔 课前小测

先别急着学,花 30 秒回答下面 3 题(答案在文末):

1. LoRA 里的「秩 $r$」是什么意思?它控制的是什么?
2. LoRA 为什么能省显存 —— 是减少了原模型的参数,还是减少了需要计算梯度的参数?
3. QLoRA 比 LoRA 多做了一步什么,让 7B 模型能在单卡 24G 显存上微调?

---

## 这节解决什么问题

上节学的 SFT 是「全参微调」—— 模型所有参数都更新。问题:**7B 模型全参微调要 100GB+ 显存**,单卡根本扛不住。一台 8 卡 A100 机器一小时几十块,普通开发者玩不起。

PEFT(Parameter-Efficient Fine-Tuning,参数高效微调)就是解决这个问题的:**只训练极少部分参数,效果接近全参微调,显存却能省 90% 以上**。其中最火的就是 LoRA,几乎所有开源大模型微调都用它。不懂 LoRA,面试基本挂。

## 核心概念 ★

### 为什么需要 PEFT

先算笔账:7B 模型全参微调的显存 = 模型权重(14GB,fp16) + 梯度(14GB) + 优化器状态(Adam 用 fp32,56GB) + 激活值 ≈ **100GB+**。单张 24G 显存的 4090 想都别想。

PEFT 的核心思路:**冻结大部分参数,只训练一小部分**。

### LoRA 的核心思想

论文 [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) 的关键洞察:微调时的权重更新 $\Delta W$ 是**低秩**的 —— 即可以用两个小矩阵的乘积来近似。

原始前向:$h = Wx$,其中 $W \in \mathbb{R}^{d \times k}$。

LoRA 把权重更新拆成两个小矩阵:

$$
h = Wx + \Delta Wx = Wx + BAx
$$

其中 $B \in \mathbb{R}^{d \times r}$,$A \in \mathbb{R}^{r \times k}$,$r \ll \min(d, k)$ 是秩。

- $W$:**冻结**,不训练,不存梯度
- $A$、$B$:**训练**,这就是 LoRA 全部可训练参数

参数量从 $d \times k$ 降到 $(d + k) \times r$。比如 $d = k = 4096$,$r = 8$:原始 $1677$ 万参数,LoRA 只有 $6.5$ 万,**降到 0.4%**。

### 缩放因子 $\alpha/r$

实际前向是:

$$
h = Wx + \frac{\alpha}{r} BAx
$$

$\alpha$ 是缩放超参。作用:**调 $\alpha$ 就能控制 LoRA 更新的强度,不用反复调学习率**。换 $r$ 时保持 $\alpha$ 不变,效果更稳。常用 $\alpha = 2r$ 或 $\alpha = 16$。

### 初始化技巧(面试必问)

> ✅ **思考一下**:如果 $A$ 和 $B$ 都随机初始化,训练开始时 $\Delta W = BA \neq 0$,会怎样?

LoRA 的初始化设计很巧妙:

- $A$:高斯随机初始化
- $B$:**零初始化**

这样训练开始时 $BA = 0$,$\Delta W = 0$,模型行为和预训练完全一致。然后梯度慢慢把 $B$ 从 0 推开,逐步学习增量。**保证从头就是好起点,不会破坏预训练模型**。

### 训练 vs 推理

- **训练时**:前向 $h = Wx + \frac{\alpha}{r}BAx$,反向只更新 $A$、$B$
- **推理时**:可以把 $BA$ **合并**回 $W$,$W_{\text{new}} = W + \frac{\alpha}{r}BA$,然后 $h = W_{\text{new}}x$

合并后**推理延迟为零** —— 和原模型一模一样。这是 LoRA 相比 Adapter、Prefix-Tuning 的大优势。

### target_modules:LoRA 挂在哪些层?

Transformer 里有多个线性层可挂 LoRA:

| 模块 | 位置 | 挂 LoRA 的效果 |
|------|------|----------------|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention 的 Q/K/V/O | **最常用**,效果/参数比最高 |
| `gate_proj`, `up_proj`, `down_proj` | FFN 的三个线性 | 效果更好,但参数翻倍 |
| `lm_head` | 输出层 | 一般不挂(收益小) |

经验:小模型(7B)只挂 attention 四层就够;大模型或难任务可把 FFN 也挂上。预算有限时,**先挂 attention,再按需扩 FFN**。

### 其他 PEFT 方法(了解)

| 方法 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **LoRA** | 低秩矩阵近似权重增量 | 效果好、推理无延迟 | 理论上不如全参 |
| **Adapter** | 在层间插入小 MLP | 参数少 | 推理多一层,有延迟 |
| **Prefix-Tuning** | 在 attention 前加可训练前缀 | 参数极少 | 难调、效果不稳 |
| **P-Tuning v2** | Prefix-Tuning 改进版 | 效果提升 | 仍不如 LoRA 稳 |

主流选择:**LoRA / QLoRA**,其他方法在面试里知道名字即可。

### QLoRA:让 7B 模型跑在单卡上

论文 [QLoRA](https://arxiv.org/abs/2305.14314) 在 LoRA 基础上多了一步:**把冻结的基座模型量化到 4-bit**(NF4 格式),只对 LoRA 参数用 fp16 训练。

显存账:7B 模型 fp16 = 14GB → NF4 量化 ≈ 3.5GB。**省下的 10GB+ 全给 LoRA 训练用**,于是 24G 单卡能跑 7B 微调。

QLoRA 三个关键技术:

1. **NF4 量化(NormalFloat 4-bit)**:针对正态分布权重设计的 4-bit 量化格式,信息损失最小
2. **Double Quantization(双重量化)**:把量化常数本身也量化,再省 0.5GB
3. **Paged Optimizer**:用统一内存处理显存峰值(优化器状态溢出时自动转 CPU),避免 OOM

效果:QLoRA 微调的 Guanaco 模型在 OpenAssistant 排行榜上达到 ChatGPT 99.3% 的水平。

## 为什么这样设计

### 为什么低秩近似有效?

直觉:预训练已经学到了「通用能力」,微调只是在这个能力上做「小幅修正」—— 这种修正不需要动所有维度,一个低秩子空间就够。论文实测:$\Delta W$ 的有效秩很低,即使 $r=8$ 也能抓住绝大部分信息。

| 对比 | 全参微调 | LoRA | QLoRA |
|------|----------|------|-------|
| **可训练参数** | 100% | ~0.1%-1% | ~0.1%-1% |
| **显存(7B)** | 100GB+ | 30GB 左右 | **10-16GB(单卡可跑)** |
| **效果损失** | —— | 几乎无损 | 轻微 |
| **推理延迟** | 基准 | **0(可合并)** | 0(可合并) |
| **适用场景** | 大厂有钱 | 主流选择 | 单卡 / 消费级 GPU |
| **训练速度** | 基准 | 略快(参数少) | 略慢(需反量化) |

### 为什么 B 初始化为 0?

训练要从一个「已知好」的点出发。预训练的 $W$ 已经是好模型,如果一开始 $\Delta W \neq 0$,等于在好模型上加了随机扰动,可能直接崩。$B=0$ 保证 $\Delta W=0$,起点干净,梯度慢慢把有用增量学出来。

### 为什么推理能合并?

LoRA 的 $BA$ 和原 $W$ 是相加关系:$W_{\text{new}} = W + \frac{\alpha}{r}BA$。加完就是一个普通矩阵,和原模型结构完全一样。合并后模型大小、推理速度都不变。Adapter 做不到这点(它是层间插入,结构变了)。

## 代码:最小实现

**① 用 numpy 手写一个 LoRA 层**

```python
import numpy as np

class LoRALayer:
    """最小 LoRA 实现:冻结 W,只训练 A、B"""
    def __init__(self, d_in, d_out, r=8, alpha=16):
        # 原始权重(冻结,不训练)
        self.W = np.random.randn(d_out, d_in) * 0.02   # 模拟预训练权重
        # LoRA 增量(可训练)
        self.A = np.random.randn(r, d_in) * 0.02        # 高斯初始化
        self.B = np.zeros((d_out, r))                   # ★ B 初始化为 0
        self.scale = alpha / r

    def forward(self, x):
        # h = Wx + (alpha/r) * BAx
        return self.W @ x + self.scale * (self.B @ self.A) @ x

    def trainable_params(self):
        return self.A.size + self.B.size   # 只有 A、B 是可训练参数

# 演示:对比参数量
d_in, d_out, r = 4096, 4096, 8
lora = LoRALayer(d_in, d_out, r=r)
print(f"原始 W 参数: {d_in * d_out:,}")              # 16,777,216
print(f"LoRA 可训练参数(A+B): {lora.trainable_params():,}")  # 65,536
print(f"参数占比: {lora.trainable_params() / (d_in * d_out) * 100:.2f}%")  # 0.39%

# 验证:初始时 ΔW = 0,输出和原模型一致
x = np.random.randn(d_in, 1)
print(f"初始 ΔW 是否为0: {np.allclose(lora.B @ lora.A, 0)}")   # True
```

**② 用 PEFT 库做 QLoRA 配置(示意)**

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# ① QLoRA 的 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 量化
    bnb_4bit_quant_type="nf4",              # NF4 格式(QLoRA 核心)
    bnb_4bit_use_double_quant=True,         # 双重量化,再省显存
    bnb_4bit_compute_dtype="bfloat16",      # 计算时反量化到 bf16
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B", quantization_config=bnb_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# ② LoRA 配置
lora_config = LoraConfig(
    r=8,                                    # 秩,常用 8/16/64
    lora_alpha=16,                          # 缩放因子,常用 2*r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 挂在 attention 上
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出:trainable params: 6,553,600 || all params: 7,621,358,592 || trainable%: 0.09%
```

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/lab/index.html) 在线试跑

## ⚠️ 易错点 / 面试陷阱

> **陷阱 1**:记反初始化 —— 「A 用高斯、B 用 0」。**正确**:A 高斯初始化(保证有信息流),B 零初始化(保证 $\Delta W = 0$,起点干净)。面试常考。

> **陷阱 2**:$r$ 越大效果越好?不一定。$r$ 太大会过拟合 + 显存涨,太小则表达能力不够。**常用 $r = 8$ 或 $16$**,特定任务最多到 64。

> **陷阱 3**:以为 LoRA 推理有额外延迟。其实**可以合并** $BA$ 回 $W$,推理速度和原模型一样。这一点是 LoRA 比 Adapter 强的关键。

> **陷阱 4**:QLoRA = LoRA + 量化基座。量化的是**冻结的基座模型**(4-bit NF4),LoRA 参数本身仍是 fp16 训练。别搞反。

> **陷阱 5**:合并 LoRA 后想再训一次?**合并后无法分离原权重和 LoRA 增量**。多 LoRA 切换场景(如一个基座服务多个任务)应**保留不合并**,推理时动态加载不同 LoRA。

## 🎯 面试会怎么考

- **八股题**:「LoRA 的原理是什么,为什么能省显存」「为什么 B 初始化为 0」「$r$ 怎么选,大了小了分别什么后果」「QLoRA 比 LoRA 多了什么」「LoRA 和 Adapter 的区别」
- **手撕题**:「用 numpy 写一个 LoRA 层的前向」「给定权重,算 LoRA 的可训练参数量」
- **深挖题**:「为什么用低秩近似有效,有理论解释吗」「QLoRA 的 NF4 量化原理是什么,为什么不用 INT4」「LoRA 能完全替代全参微调吗,什么场景下 LoRA 不够」「推理时怎么合并,合并条件是什么」

## 📂 简历可写的项目

**单卡 QLoRA 微调 7B 模型** —— 用 Qwen2-7B 或 Llama-3-8B 做基座,用 QLoRA 在单张 24G 消费级 GPU(4090/3090)上微调中文指令遵循任务,报告:显存占用、训练时间、可训练参数占比、微调前后效果对比(用 C-Eval / CMMLU 等中文基准)。
> 项目名:「基于 QLoRA 的 Qwen2-7B 中文指令对齐(单卡 24G)」,技术栈:transformers + peft + bitsandbytes。

## 🚀 挑战

固定一个数据集和基座模型,对比 LoRA 的 $r = 4, 8, 16, 64$ 四个设置下的:① 可训练参数量 ② 训练显存 ③ 评测分数。画出「$r$ vs 效果」和「$r$ vs 显存」两条曲线,找性价比最高的 $r$。思考:$r$ 在什么范围再增大也没收益了?(这是工业界调 LoRA 的核心实验)

## 🔗 延伸阅读

- 📄 [LoRA 原论文](https://arxiv.org/abs/2106.09685) —— 低秩适配的开山之作,必读
- 📄 [QLoRA 论文](https://arxiv.org/abs/2305.14314) —— 单卡微调 7B 的关键技术,NF4 量化原理
- 💻 [PEFT 官方库](https://github.com/huggingface/peft) —— HuggingFace 官方 PEFT 实现,LoRA/QLoRA/Adapter 都支持

---

## ✅ 课后小测(答案)

**课前小测答案:**

1. **秩 $r$ 控制的是 LoRA 增量矩阵 $\Delta W$ 的「信息容量」**。$\Delta W = BA$,$B$ 是 $d \times r$,$A$ 是 $r \times k$,$r$ 越大表达能力越强但参数越多。常用 $r=8$ 或 $16$。

2. **减少的是「需要计算梯度和优化器状态的参数」**。原模型权重 $W$ 冻结,不存梯度不存优化器状态(这是显存大头),只对 A、B 训练。所以显存省在反向传播和优化器,而不是前向。可训练参数降到 0.1%-1%,但原模型还是要载入显存做前向(这正是 QLoRA 进一步要解决的)。

3. **QLoRA 把冻结的基座模型量化到 4-bit(NF4 格式)**。基座从 14GB(fp16) 降到 3.5GB(NF4),省下的显存给 LoRA 训练用,于是 7B 模型能在单卡 24G 上微调。配合双重量化(Double Quantization)和 Paged Optimizer 进一步省显存。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——
> 写出 LoRA 的前向公式,解释 B 为什么初始化为 0、推理时为什么可以无延迟,并说出 QLoRA 在 LoRA 基础上多了哪三步。
> (答案:$h = Wx + \frac{\alpha}{r}BAx$;B=0 保证起点 $\Delta W=0$ 不破坏预训练;推理合并 $W_{\text{new}}=W+\frac{\alpha}{r}BA$ 无延迟;QLoRA 多了 NF4 量化 + 双重量化 + Paged Optimizer)
