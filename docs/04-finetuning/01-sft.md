# 01 · 监督微调 SFT:让模型学会听话

> 一句话:**学完这节,你能回答「SFT 和预训练的损失函数差在哪、为什么 LIMA 用 1000 条数据就够了」这两个面试高频题。**

## 🤔 课前小测

先别急着学,花 30 秒回答下面 3 题(答案在文末):

1. 预训练时模型对所有 token 算 loss,SFT 时只在「回答部分」算 loss —— 为什么 prompt 部分不算?
2. 直觉判断:用 10 万条普通客服话术 vs 1000 条精心标注的指令数据微调同一个模型,哪个效果更好?
3. SFT 后模型把预训练学的能力「忘了」(比如英文变差),这个现象叫什么?怎么缓解?

---

## 这节解决什么问题

预训练完的大模型其实**不会回答问题**,它只会「续写」。你问它「中国的首都是哪」,它可能续写成「中国的首都是哪?这是一个常见问题,下面我们来详细介绍……」—— 它以为你在让它写作文。

SFT(Supervised Fine-Tuning,监督微调)就是教会模型:**看到指令 → 给出回答**,而不是无脑续写。这是把一个「语言模型」变成「对话助手」的关键一步。不懂 SFT,你做不了任何垂直领域大模型应用。

## 核心概念 ★

### 从「补全」到「遵循指令」

预训练和 SFT 用的都是 **next-token prediction**(预测下一个 token),但目标完全不同:

| 维度 | 预训练 | SFT |
|------|--------|-----|
| **目标** | 学语言规律 | 学会遵循指令 |
| **数据** | 海量无标注文本(万亿 token) | 少量「指令-回答」对(几千~几十万条) |
| **Loss 范围** | 所有 token | **只在回答部分**算 loss |
| **学习率** | 较大($\sim 10^{-4}$) | 小($\sim 10^{-5}$,小 1-2 个数量级) |
| **训练轮数** | 1 轮左右 | 2-5 轮(epoch) |

### SFT 的损失函数

核心区别:**loss 只在回答 token 上算,prompt 部分 mask 掉**。

$$
L_{\text{SFT}} = -\sum_{t=1}^{T} m_t \cdot \log P(y_t \mid y_{<t}, x)
$$

其中 $m_t$ 是 mask:$m_t = 1$ 表示第 $t$ 个 token 属于回答部分(算 loss),$m_t = 0$ 表示属于 prompt(不算 loss)。$x$ 是指令,$y_t$ 是回答的第 $t$ 个 token。

为什么这样?因为我们要奖励模型「生成好的回答」,而不是「背诵好的 prompt」。prompt 是用户给的,模型不用学怎么生成它。

### 指令数据格式

标准的 SFT 数据长这样:

```json
{
  "instruction": "把下面句子翻译成英文",
  "input": "今天天气真好",
  "output": "The weather is nice today."
}
```

拼成模型输入(用特殊分隔符):

```
### Instruction:
把下面句子翻译成英文
### Input:
今天天气真好
### Response:
The weather is nice today.
```

只有 `The weather is nice today.` 这部分参与 loss 计算。

### 数据质量 > 数量:LIMA 的证明

> ✅ **思考一下**:如果 1000 条高质量数据就能微调出好模型,那那些搞百万条数据的公司在干嘛?

[LIMA 论文](https://arxiv.org/abs/2305.11206) 做了个震撼实验:**只用 1000 条**人工精写的指令数据微调 LLaMA,效果就能和 GPT-3.5 接近。这说明:

- SFT 的本质是**「激活」**模型在预训练中已经学到的能力,而不是灌输新知识
- 数据的**格式规范、多样性、回答质量**远比数量重要
- 10 万条脏数据不如 1 万条干净数据

**为什么 SFT 能「激活」而不是「灌输」?** 预训练已经看了万亿 token 的文本,模型内部其实已经「知道」怎么翻译、怎么写代码、怎么总结。它只是不知道**什么时候该用哪种能力**。SFT 的指令格式相当于告诉模型:「看到这种格式的输入,你就调用 X 能力」。所以 SFT 用很少的数据就能见效。

### SFT 数据怎么清洗(面试常问)

工业界 SFT 数据清洗的关键步骤:

| 步骤 | 做什么 | 去除什么 |
|------|--------|----------|
| 去重 | 近似哈希(MinHash) | 重复样本 |
| 格式校验 | 检查 instruction/input/output 完整 | 空字段、结构错 |
| 长度过滤 | 去掉过短(回答<5字)/过长 | 噪声 / 截断 |
| 质量打分 | 用强模型(如 GPT-4)给回答打分 | 低分回答 |
| 多样性 | 按任务类型 / 主题聚类平衡 | 单一类型堆砌 |
| 安全过滤 | 有害、涉政、PII(个人信息) | 合规风险 |

经验:**清洗掉 30%-50% 的脏数据后,模型效果反而提升**。这就是「质量 > 数量」的实证。

### 灾难性遗忘(Catastrophic Forgetting)

SFT 后模型可能「变笨」—— 比如英文能力下降、数学不会做了。原因:SFT 数据分布和预训练差太多,模型参数被带偏了。

缓解方法:

| 方法 | 做法 | 代价 |
|------|------|------|
| **小学习率** | SFT lr 比预训练小 1-2 个数量级 | 训练慢 |
| **混入预训练数据** | SFT 时混入部分原始预训练文本 | 增加数据准备成本 |
| **少训几轮** | 2-3 epoch 就够,别过头 | —— |
| **用 LoRA** | 冻结原参数,只训增量(下节讲) | 表达能力略受限 |

### SFT / 指令调优 / RLHF 的关系

| 概念 | 说明 |
|------|------|
| **指令调优(Instruction Tuning)** | SFT 的一种,用指令格式数据微调 |
| **SFT** | 监督微调,范围更广(包括但不仅限于指令) |
| **RLHF** | SFT **之后**的对齐阶段,用人类偏好做强化学习(下一节讲) |

标准 pipeline:**预训练 → SFT → RLHF/DPO**。

## 为什么这样设计

### 为什么 loss 只在回答部分算?

想象你是老师教学生答题。你会盯着学生**写的答案**打分,不会因为他**抄错了题目**就扣分 —— 题目是你出的。同理,prompt 是用户给的,模型不应该为「生成 prompt」负责,只为「生成回答」负责。

如果 prompt 也算 loss,模型会浪费容量去学「怎么复述指令」,反而降低回答质量。

### 为什么学习率要小?

预训练后的模型处于一个**精细平衡**的状态(万亿 token 训出来的)。SFT 是在这个基础上做「微调」—— 只需轻轻拨一下,让它学会指令格式。学习率一大,就把预训练学的东西冲垮了(灾难性遗忘)。

| 对比 | 高质量少量(1000 条) | 低质量海量(10 万条) |
|------|----------------------|----------------------|
| 训练速度 | 快 | 慢 |
| 遗忘风险 | 低 | 高 |
| 效果 | 格式规范、能力保留好 | 容易学坏、答非所问 |
| 成本 | 标注贵 | 标注便宜 |

## 代码:最小实现

**① SFT 数据处理:区分 prompt 和 response 的 mask**

```python
import torch

def build_sft_batch(tokenizer, instruction, input_text, output):
    """构造一条 SFT 样本:prompt 部分 label=-100(不算loss),response 部分算 loss"""
    # 1. 拼 prompt
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response:\n"

    # 2. 分别编码
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(output, add_special_tokens=False) + [tokenizer.eos_token_id]

    # 3. 拼成完整序列
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids   # -100 是 PyTorch CrossEntropy 的忽略标记
    return {"input_ids": input_ids, "labels": labels}

# 玩具演示
class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split()  # 简化:用空格分词
    eos_token_id = -1

tok = FakeTokenizer()
sample = build_sft_batch(tok, "翻译成英文", "你好", "Hello world.")
print("input_ids:", sample["input_ids"])
print("labels:   ", sample["labels"])
# labels 里 prompt 部分是 -100,只有 response 部分有真实 token —— 这就是 SFT 的精髓
```

**② 用 transformers Trainer 写 SFT 核心配置(示意)**

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# 关键:DataCollatorForSeq2Seq 会自动按 labels=-100 做 mask,并 pad 到等长
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,   # prompt 位置不算 loss
    padding=True
)

training_args = TrainingArguments(
    output_dir="./sft-output",
    num_train_epochs=3,                # SFT 通常 2-5 epoch,别太多
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,     # 小显存用梯度累积凑大 batch
    learning_rate=2e-5,                # ★ 关键:比预训练小 1-2 个数量级
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                          # 混合精度省显存
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,          # 已构造好 input_ids + labels
    data_collator=data_collator,
)
trainer.train()
```

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/lab/index.html) 在线试跑

## ⚠️ 易错点 / 面试陷阱

> **陷阱 1**:SFT 学习率用预训练的值($10^{-4}$) → 灾难性遗忘,模型变傻。**正确**:用 $10^{-5}$ 量级,小 1-2 个数量级。

> **陷阱 2**:把 prompt 部分也算进 loss → 模型学会「复述指令」,回答质量反而下降。**正确**:prompt token 的 label 设成 -100。

> **陷阱 3**:觉得「数据越多越好」,堆百万条脏数据 → 模型学坏、格式混乱。**正确**:优先保证质量,LIMA 证明 1000 条精标数据就能有好效果。先小规模验证,再扩量。

## 🎯 面试会怎么考

- **八股题**:「SFT 的目标是什么,和预训练有什么区别」「灾难性遗忘是什么,怎么解决」「SFT 的 loss 怎么算,为什么 prompt 部分不算 loss」「SFT 和 RLHF 是什么关系」「SFT 学习率为什么要小」
- **手撕题**:「写一个函数,把 instruction/input/output 转成带 mask 的 input_ids 和 labels」「给定一个 batch,手算 SFT 的 loss」
- **深挖题**:「SFT 数据怎么清洗?多少数据够用?」「学习率怎么选?epoch 怎么定?」「为什么 LIMA 用 1000 条数据就够了,你怎么解释」「SFT 训多少 epoch 会过拟合,怎么判断」

## 📂 简历可写的项目

**垂直领域 SFT 助手** —— 选一个垂直领域(电商客服 / 医疗问答 / 法律咨询 / 代码助手),用 Qwen-7B 或 Llama-3-8B 做基座,构造 5000-10000 条高质量指令数据,做 SFT 微调,对比 SFT 前后在领域问答上的效果(用 BLEU / Rouge / 人工评分)。
> 项目名:「基于 Qwen-7B SFT 的电商客服对话助手」,技术栈:transformers + peft + 自建指令数据集。

## 🚀 挑战

手动构造 100 条高质量指令数据(覆盖翻译、摘要、问答、代码、写作 5 类任务,每类 20 条),用 transformers + LoRA(下节学)微调一个 Qwen-1.8B。观察:微调前后模型回答的**格式规范性**和**指令遵循能力**有什么变化?再把数据量扩到 500 条,效果提升明显吗?(这会帮你理解「数据质量 > 数量」)

## 🔗 延伸阅读

- 📄 [LIMA 论文:Less Is More for Alignment](https://arxiv.org/abs/2305.11206) —— 1000 条数据微调的里程碑,理解「质量 > 数量」
- 📄 [Alpaca 论文](https://crfm.stanford.edu/2023/03/13/alpaca.html) —— $600 复刻 GPT-3.5,SFT 数据构造的经典范例
- 💻 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) —— 开箱即用的 SFT 训练框架,支持几十种模型,适合上手实操

---

## ✅ 课后小测(答案)

**课前小测答案:**

1. **因为 prompt 是用户给的,模型不需要学「怎么生成 prompt」**。只对回答部分算 loss,能引导模型把容量用在「生成好的回答」上,而不是浪费在复述指令上。实现上把 prompt token 的 label 设成 -100(PyTorch 自动忽略)。

2. **1000 条精心标注的数据效果更好**。LIMA 论文证明 1000 条高质量数据就能接近 GPT-3.5。SFT 的本质是「激活」预训练已有的能力,不是灌输新知识。脏数据会让模型学坏(格式乱、答非所问、价值观偏差)。

3. **灾难性遗忘(Catastrophic Forgetting)**。缓解:① 小学习率(比预训练小 1-2 个数量级);② 混入部分预训练数据;③ 少训几轮(2-3 epoch);④ 用 LoRA 冻结原参数。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——
> 给定一条 SFT 数据 `{"instruction":"求和","input":"3+5","output":"8"}`,写出拼接后的完整输入序列,并标出哪些位置的 label 是 -100、哪些是真实 token。再说出 SFT 的学习率为什么要比预训练小。
> (答案:Response 之前的 token label=-100,Response 部分(包含 EOS)label 是真实 token。学习率小是为了不破坏预训练学到的精细平衡,避免灾难性遗忘)
