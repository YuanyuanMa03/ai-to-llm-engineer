# 第 3 阶段 · 预训练

> 目标:理解 LLM 是怎么「炼」出来的,知道钱花在哪里、为什么。

## 这个阶段解决什么

面试官常问「你了解预训练吗」,多数人只会背「大规模语料 + 自监督」。本阶段拆解预训练的三个支柱:**分词、架构、数据配比**,让你能说清「为什么用 BPE」「为什么 Decoder-only 成主流」「Chinchilla 定律告诉你 10B 模型该用多少数据」。

## 学习目标

- [ ] 理解 BPE / WordPiece / SentencePiece 的区别与权衡
- [ ] 说清 Decoder-only vs Encoder-only vs Encoder-Decoder 的适用场景
- [ ] 理解 MoE(混合专家)为什么能省算力
- [ ] 能用 Scaling Laws 估算「模型大小 vs 数据量 vs 算力」
- [ ] 知道预训练数据清洗的基本流程

## 本阶段章节

- [Tokenizer:BPE / SentencePiece](01-tokenizer.md)
- [LLM 架构:Decoder-only / MoE](02-architecture.md)
- [Scaling Laws 与数据配比](03-scaling-laws.md)

## 🎯 面试高频考点(本阶段)

| 考点 | 难度 | 说明 |
|------|------|------|
| BPE 原理 | ★★★ | 分词高频题 |
| Decoder-only 为什么成主流 | ★★ | 训练效率 + Scaling |
| RoPE 旋转位置编码 | ★★★ | 进阶必问 |
| Chinchilla 最优数据配比 | ★★ | token : 参数 ≈ 20:1 |
| MoE 路由机制 | ★★ | 稀疏激活 |
| RMSNorm vs LayerNorm | ★★ | 去 centering |

## 📂 简历项目建议

- **训练一个 character-level 小语言模型**(几十万参数),观察 loss 下降
- **对比不同 Tokenizer** 在中文上的切分效果

---

> ⏳ 正文编写中。本导览页已就绪,子页面内容将逐步填充。
