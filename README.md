# 📘 AI 手帐 · 从小白到大模型算法工程师

> 不只记录「它是什么」,更要记录「**面试会怎么考、简历能写什么项目**」。

这是一份**求职导向**的大模型算法工程师学习路线图。每一节都把知识、代码、面试考点、简历项目串成一条线。

🌐 **在线阅读**:[yuanyuanma03.github.io/ai-to-llm-engineer](https://yuanyuanma03.github.io/ai-to-llm-engineer/)

![AI 知识总览](assets/sketchnote-overview.png)
> 手绘图来自 [microsoft/AI-For-Beginners](https://github.com/microsoft/AI-For-Beginners) by Tomomi Imura,MIT 协议

## 🗺️ 全链路路线图

15 篇正文 · 5 个阶段 · 每篇含课前课后测、可跑代码、面试考点、简历项目。

```
数学 & 编程基础  →  深度学习 & Transformer  →  预训练
                                                        ↓
                  应用 & 部署  ←  微调 & 对齐
```

| 阶段 | 主题 | 核心产出 | 预估时长 |
|------|------|----------|----------|
| **01** | [数学与编程基础](01-fundamentals/README.md) | 能手算梯度、写 numpy 实现 | 2-3 周 |
| **02** | [深度学习与 Transformer](02-deep-learning/README.md) | 能从零写 Attention | 3-4 周 |
| **03** | [预训练](03-pretraining/README.md) | 理解 LLM 怎么炼出来的 | 2-3 周 |
| **04** | [微调与对齐](04-finetuning/README.md) | 能用 LoRA 微调一个模型 | 3-4 周 |
| **05** | [应用与部署](05-deployment/README.md) | 能搭 RAG / 调 vLLM | 2-3 周 |
| **06** | [前沿专题](06-frontier/README.md) | 跟上 MCP 等 2026 新趋势 | 持续 |

## 🧰 求职工具箱

不只是看教程,还能直接练:

| 工具 | 说明 |
|------|------|
| 🎯 **[面试自测器](../quiz/)** | 30 道真题随机抽,按阶段/难度筛选,先想答案再翻看 |
| ✋ **[手撕代码题合集](hand-coding.md)** | Attention / Softmax / LoRA 等 10 道高频手写题,含参考实现 |
| 💬 **[社区讨论区](https://github.com/YuanyuanMa03/ai-to-llm-engineer/discussions)** | 投稿你遇到的面试题,讨论疑难 |

## 🎯 为什么又是另一个教程?

市面上高星的 LLM 教程已经很多了,但它们大多是:

- ❌ **资源链接农场**(一堆链接,自己得去啃)
- ❌ **纯知识体系**(不告诉你面试怎么考、简历写什么)
- ❌ **工程参考手册**(面向已入门的人,小白看不懂)

这份教程的差异化:**每一节都点明「面试考点 + 简历项目」**,把零散知识变成求职弹药。

## 📐 每一节的固定结构

每一篇教程都按统一模板写,融合了微软 AI-For-Beginners 的教学法 + 求职导向:

- **🤔 课前小测** → 激活先验知识
- **核心概念 ★** → 配 KaTeX 公式 + ✅思考提示
- **为什么这样设计** → 直觉解释 + 对比表格
- **代码:最小实现** → 可直接跑的 Python
- **⚠️ 易错点 / 面试陷阱** → 高频踩坑
- **🎯 面试会怎么考** → 八股 + 手撕 + 深挖
- **📂 简历可写的项目** → mini 项目建议
- **🚀 挑战** → 开放动手任务
- **✅ 课后小测** → 前后测闭环,检验掌握

## 🚀 从哪里开始

- **完全小白**:从 [第 1 阶段 · 数学基础](01-fundamentals/README.md) 开始,按顺序走
- **有 ML 基础**:直接跳 [第 2 阶段 · Transformer](02-deep-learning/03-transformer.md)
- **想冲面试**:重点看每节的「🎯 面试会怎么考」和「📂 简历项目」

## 🛠️ 技术栈

- 教程渲染:[Docsify](https://docsify.js.org/) —— 零构建,运行时渲染 Markdown
- 数学公式:KaTeX
- 部署:GitHub Pages

## 📝 如何贡献

欢迎提 Issue 和 PR:

- 发现内容错误 → 每页底部有「在 GitHub 上编辑本页」链接
- 想补充某节内容 → 提 PR,请遵循 [写作模板](TEMPLATE.md)
- 有面试题想分享 → 提 Issue,标注「面试题投稿」

## 📄 许可

MIT License
