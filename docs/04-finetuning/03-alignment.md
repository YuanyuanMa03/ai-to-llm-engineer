# 03 · 对齐:RLHF 与 DPO

> 一句话:**学完这节,你能回答「RLHF 为什么要奖励模型、DPO 为什么能跳过 RM、对齐税是什么」这三个面试必考题。**

## 🤔 课前小测

先别急着学,花 30 秒回答下面 3 题(答案在文末):

1. SFT 之后的模型,为什么还需要 RLHF?SFT 教的不是「怎么回答」吗?
2. DPO 比 RLHF 省掉了哪一步?为什么这一步能省?
3. 「对齐税」是什么意思?为什么对齐之后模型反而可能变笨?

---

## 这节解决什么问题

SFT 教会了模型「听指令回答」,但 SFT 有个硬伤:**它学的是「模仿数据」,无法区分好回答和坏回答**。比如训练数据里有一个粗鲁的回答,模型也会学粗鲁;面对有害提问,SFT 模型可能照答不误。

**对齐(Alignment)** 就是要让模型的输出**符合人类偏好**:有用(Helpful)、诚实(Honest)、无害(Harmless)。RLHF 和 DPO 是当前两大主流对齐方法。ChatGPT 横空出世靠的就是 RLHF,几乎所有头部大厂模型都过了这一步。对齐是大模型工程的「最后一公里」,也是面试区分度高的话题。

## 核心概念 ★

### 为什么需要对齐

| 模型阶段 | 会什么 | 还缺什么 |
|----------|--------|----------|
| 预训练 | 续写文本 | 不会回答问题 |
| **SFT 后** | 听指令、会回答 | 不会区分好坏、可能有害、可能谎报 |
| **对齐后** | 有用、诚实、无害 | —— |

举例:用户问「怎么黑掉别人的 wifi」,SFT 模型可能直接给步骤(模仿了网上教程),对齐后的模型会拒绝并解释为什么不合适。

### RLHF:三阶段对齐

RLHF(Reinforcement Learning from Human Feedback)来自 [InstructGPT 论文](https://arxiv.org/abs/2203.02155),分三步:

**阶段 1:SFT** —— 上节学的,先让模型会回答。

**阶段 2:训练奖励模型(Reward Model,RM)**

- 让人类标注**偏好对**:对同一个 prompt,给出两个回答 A 和 B,人工标注「A 比 B 好」(或反之)
- RM 学一个打分函数 $r_\phi(x, y)$:给定 prompt $x$ 和回答 $y$,输出一个标量分数
- RM 的训练目标(Bradley-Terry 模型):

$$
L_{\text{RM}} = -\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

其中 $y_w$ 是更好的回答($w$ = win),$y_l$ 是更差的回答($l$ = lose),$\sigma$ 是 sigmoid。直觉:让好回答的分数比坏回答高。

**阶段 3:用 PPO 优化策略**

- 把 SFT 模型作为初始策略 $\pi_\theta$
- 用强化学习让 $\pi_\theta$ 生成 RM 分数高的回答
- 但有个坑:模型为了刷高分可能输出乱码(只讨好 RM),所以要加 **KL 惩罚** —— 让 $\pi_\theta$ 不要离参考模型 $\pi_{\text{ref}}$(通常是 SFT 模型)太远

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta}[r_\phi(x, y)] - \beta \, \text{KL}(\pi_\theta(\cdot|x) \, \| \, \pi_{\text{ref}}(\cdot|x))
$$

$\beta$ 是 KL 惩罚强度:太大 = 模型不动(和 SFT 一样),太小 = 模型乱跑(刷分崩坏)。

PPO 的 clip 目标:

$$
L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t \hat{A}_t, \, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是重要性采样比,$\hat{A}_t$ 是优势函数,$\epsilon$ 控制策略更新幅度。

### DPO:跳过 RM 的直接偏好优化

> ✅ **思考一下**:RLHF 要训 RM、要调 PPO、要稳 KL —— 太复杂。能不能直接用偏好对优化模型,跳过 RM?

[DPO 论文](https://arxiv.org/abs/2310.16944)(Direct Preference Optimization)的核心洞察:**RLHF 的目标函数可以解析地重写,最优策略直接由偏好数据表示,不需要显式训 RM**。

DPO 损失:

$$
L_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

直觉:让模型**相对参考模型**,提高好回答 $y_w$ 的概率、降低坏回答 $y_l$ 的概率。

- $\pi_\theta$:正在训练的策略模型
- $\pi_{\text{ref}}$:参考模型(通常是 SFT 模型,冻结)
- $\beta$:温度参数,大 = 保守(贴近 ref),小 = 激进(偏离 ref)
- $y_w, y_l$:偏好对里的好、坏回答

DPO 用的是分类 loss(sigmoid 形式),**梯度稳定,实现简单**,一个 Transformer + 二分类 loss 就行,不用 PPO。

### RLAIF:用 AI 代替人类标注

[Constitutional AI](https://arxiv.org/abs/2212.08073)(Claude 用的方法):让一个强模型(如 GPT-4)代替人类标注偏好。流程:

1. 给 AI 一份「宪法」(Constitution,原则清单)
2. AI 按宪法评估两个回答哪个好
3. 用 AI 标注的偏好数据训 RM 或直接 DPO

好处:**标注成本暴降、规模可放大**。坏处:AI 标注有偏见,可能学到「AI 味」而非真正人类偏好。

### Reward Hacking(RLHF 的致命坑)

RLHF 阶段 3 用 PPO 优化策略时,模型可能找到 RM 的漏洞 —— **输出能骗高分的乱码**。这叫 reward hacking(奖励欺骗)。

典型表现:模型输出「The answer is. The answer is. The answer is...」这种重复句式,RM 给高分(因为训练数据里好回答常带「The answer is」),但人类看是垃圾。

这就是为什么 RLHF 必须加 **KL 惩罚**:限制策略不能离 SFT 模型太远,防止刷分崩坏。DPO 天然没这个问题(直接用偏好对,没有可被欺骗的 RM)。

### 2024 新趋势:GRPO

DeepSeek 在 [DeepSeekMath / R1](https://arxiv.org/abs/2401.02966) 里用 GRPO(Group Relative Policy Optimization)替代 PPO,核心改进:**去掉 value model**,用同一 prompt 的多个采样回答的**组内相对优势**代替绝对优势。显存更省、更稳。R1 的强推理能力主要靠 GRPO 训出来,是当前对齐研究的热点。

## 为什么这样设计

### 为什么 RLHF 要 RM,而 DPO 不要?

RLHF 的逻辑链:人类偏好 → RM(把偏好变成可微分数)→ PPO(用分数优化策略)。RM 是中间桥梁,因为 PPO 需要连续可微的奖励信号。

DPO 的突破:数学上证明,**RLHF 的最优解可以不显式构造 RM,直接从偏好对推导出来**。RM 被消融掉了,偏好对直接驱动策略。代价是 DPO 没有 RM 那个「可泛化的奖励信号」,只能基于现有偏好对优化。

| 对比 | RLHF | DPO |
|------|------|-----|
| **需要训 RM** | 是 | **否** |
| **训练方式** | 强化学习(PPO) | 监督学习(分类 loss) |
| **稳定性** | 难调,容易崩 | **稳定** |
| **实现复杂度** | 高(4 个模型:策略、价值、RM、ref) | 低(2 个模型:策略、ref) |
| **显存** | 大 | **小** |
| **在线探索** | 支持(PPO 在线采样) | 不支持(离线数据) |
| **效果上限** | 略高(可探索) | 略低(受数据限制) |

### 为什么 KL 惩罚 / DPO 要带 ref 模型?

> **陷阱**:如果不加 ref 约束,策略会过度优化 RM 或偏好对,输出乱码。

约束的目的:**让模型在「对齐偏好」和「保持语言能力」之间平衡**。KL 项 / DPO 里的 $\log(\pi_\theta/\pi_{\text{ref}})$ 都是同一个意思:别离 SFT 模型太远。$\beta$ 控制松紧。

### 为什么 $\beta$ 是温度参数?

- $\beta$ 大 → loss 里 ref 项权重大 → 模型保守,变化小(可能对齐不够)
- $\beta$ 小 → 模型激进,大幅偏离 ref(可能崩坏,失去语言能力)
- 常用 $\beta = 0.1$ 左右

### 对齐税

对齐后模型在通用能力(数学、代码、推理)上可能**下降** —— 这叫「对齐税」。原因:对齐数据偏向「安全 / 礼貌 / 拒绝」,模型学会了过度保守。

缓解:① 对齐数据里混入能力数据;② 用 DPO 替代 RLHF(对齐税更小);③ 对齐后做能力回测,必要时重新 SFT 部分能力。

## 代码:最小实现

**① RM 损失(RLHF 阶段 2)**

```python
import torch
import torch.nn.functional as F

def reward_model_loss(r_w, r_l):
    """RM 损失:让好回答分数 r_w 比坏回答分数 r_l 高
    r_w, r_l: RM 对好/坏回答的打分(B,)
    """
    return -F.logsigmoid(r_w - r_l).mean()   # Bradley-Terry 模型

# 演示
r_w = torch.tensor([2.1, 1.5, 0.8])   # 好回答的分数
r_l = torch.tensor([0.3, 1.2, 1.0])   # 坏回答的分数
print(f"RM loss: {reward_model_loss(r_w, r_l):.4f}")   # 分数差越大 loss 越小
```

**② DPO 损失(DPO 核心)**

```python
def dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    """DPO 损失
    policy_logp_w/l: 策略模型对好/坏回答的 log概率
    ref_logp_w/l:    参考模型对好/坏回答的 log概率
    """
    # 相对参考模型的 log ratio
    logits = beta * (policy_logp_w - ref_logp_w) - beta * (policy_logp_l - ref_logp_l)
    return -F.logsigmoid(logits).mean()

# 演示:假设当前策略对好回答概率比 ref 高,对坏回答比 ref 低
policy_logp_w = torch.tensor([-2.0, -1.5, -3.0])   # 策略给好回答的 logp
policy_logp_l = torch.tensor([-4.0, -5.0, -6.0])   # 策略给坏回答的 logp(更低 = 已抑制)
ref_logp_w    = torch.tensor([-2.5, -2.0, -3.0])   # 参考模型冻结
ref_logp_l    = torch.tensor([-2.5, -2.0, -3.0])
loss = dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta=0.1)
print(f"DPO loss: {loss:.4f}")   # 模型若已正确区分好坏,loss 接近 0
```

**③ PPO clip 目标(RLHF 阶段 3,示意)**

```python
def ppo_clip_objective(ratio, advantage, epsilon=0.2):
    """PPO 的 clip 目标(要最大化)
    ratio: π_θ(a|s) / π_old(a|s),  advantage: Â
    """
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    return torch.min(surr1, surr2).mean()   # 取 min 防止策略更新过大

ratio = torch.tensor([1.1, 0.9, 1.3, 0.5])
adv   = torch.tensor([1.0, -0.5, 0.8, 1.2])
print(f"PPO objective: {ppo_clip_objective(ratio, adv):.4f}")
```

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/lab/index.html) 在线试跑

## ⚠️ 易错点 / 面试陷阱

> **陷阱 1**:以为 RLHF 很好调。**实际 PPO 极不稳**:学习率、KL 系数、batch size 任何一个没调好,模型就崩(输出重复、乱码、刷 RM 分)。面试常被追问「PPO 为什么不稳」。DPO 就是为了避开这个坑。

> **陷阱 2**:DPO 的 $\beta$ 设很小 = 激进 = 模型可能崩坏(失去语言能力,输出乱码)。$\beta$ 很大 = 保守 = 对齐效果弱。**常用 $\beta = 0.1$**,需要根据任务调。

> **陷阱 3**:对齐税(对齐后能力下降)是真实存在的。GPT-4 报告里都承认对齐后部分 benchmark 分数下降。面试问「为什么对齐后模型可能变笨」,答案:对齐数据偏向安全/拒绝,模型过度保守。

> **陷阱 4**:DPO 不是「DPO 比 RLHF 好」那么简单。RLHF 在**在线探索**(PPO 实时采样新回答)上有优势,效果上限略高;DPO 简单稳定但受离线偏好数据限制。两者各有适用场景。

## 🎯 面试会怎么考

- **八股题**:「RLHF 的三个阶段是什么」「DPO 为什么能跳过 RM」「对齐的目标是什么(Helpful/Honest/Harmless)」「对齐税是什么,怎么缓解」「KL 惩罚的作用是什么,$\beta$ 大了好还是小了好」「reward hacking 是什么」
- **手撕题**:「写 DPO 的 loss」「写 RM 的 Bradley-Terry loss」「写 PPO 的 clip 目标」
- **深挖题**:「PPO 为什么不稳定,具体哪些因素」「RLAIF 可行吗,有什么风险」「RLHF 和 DPO 谁效果上限更高,为什么」「DPO 里的 $\beta$ 怎么选,对结果的影响」「GRPO 和 PPO 的区别,为什么 DeepSeek 用 GRPO」

## 📂 简历可写的项目

**小模型 DPO 对齐实验** —— 选一个 1.5B-3B 小模型(Qwen-1.5B / Llama-3.2-1B),先 SFT,再用公开偏好数据集(如 [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) 或 [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback))做 DPO,对比 SFT 模型 vs DPO 模型在「安全性 / 有用性 / 拒绝率」上的差异。
> 项目名:「基于 DPO 的 Qwen-1.5B 偏好对齐实验」,技术栈:transformers + trl + 自建偏好数据集。

## 🚀 挑战

固定一个 SFT 模型和测试集,分别对比 SFT 模型、RLHF 模型(可用 trl 库的 PPOTrainer)、DPO 模型的输出差异。针对 3 类 prompt(有害请求、正常提问、边界模糊请求),观察三个模型的回答风格,记录:① 拒绝率 ② 回答质量 ③ 是否「过度保守」。思考:哪种方法在「安全 vs 有用」的权衡上最好?(这是 Anthropic、OpenAI 都在做的核心实验)

## 🔗 延伸阅读

- 📄 [InstructGPT 论文(RLHF 开山)](https://arxiv.org/abs/2203.02155) —— RLHF 三阶段完整流程,必读
- 📄 [DPO 论文](https://arxiv.org/abs/2310.16944) —— 跳过 RM 的数学推导,简化对齐
- 💻 [TRL 库](https://github.com/huggingface/trl) —— HuggingFace 官方对齐训练库,RLHF / DPO / PPO 都支持

---

## ✅ 课后小测(答案)

**课前小测答案:**

1. **SFT 只教模型「模仿数据」,无法区分好坏**。SFT 数据里如果有粗鲁/有害/错误回答,模型照学;面对有害提问 SFT 可能照答。对齐(RLHF/DPO)用**人类偏好**做信号,让模型学会「哪个回答更好」,从而输出有用、诚实、无害的内容。SFT 解决「会不会答」,对齐解决「答得好不好、该不该答」。

2. **DPO 跳过了「训练奖励模型 RM」这一步(以及 PPO 强化学习)**。DPO 数学上证明:RLHF 的最优策略可以直接从偏好对推导出来,$L_{\text{DPO}}$ 直接用偏好对 $(y_w, y_l)$ 优化策略,不需要中间的 RM。省掉了训 RM 的成本和 PPO 的不稳定性。

3. **对齐税 = 对齐后模型通用能力(数学、代码、推理)下降**。原因:对齐数据偏向「安全 / 礼貌 / 拒绝」,模型学会了过度保守,该答的也不答或答得差。缓解:① 对齐数据混入能力数据;② 用 DPO 替代 RLHF(对齐税更小);③ 对齐后做能力回测,必要时补 SFT。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——
> 写出 DPO 的 loss 公式,解释每一项含义;再说出 RLHF 三个阶段,以及 DPO 为什么比 RLHF 稳定。
> (答案:$L_{\text{DPO}}=-\log\sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$。$\pi_\theta$ 策略,$\pi_{\text{ref}}$ 参考,$\beta$ 温度,$y_w/y_l$ 好/坏回答。RLHF 三阶段:SFT → 训 RM → PPO 优化。DPO 稳定是因为用分类 loss 替代了 PPO 强化学习,梯度稳定)
