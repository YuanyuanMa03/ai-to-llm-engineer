# 第 5 阶段 · 应用与部署

> 目标:能搭一个 RAG 系统,会调 vLLM 做推理加速,理解 Agent 怎么调工具。

## 这个阶段解决什么

模型训出来只是开始,真正落地要解决**推理慢、幻觉、知识过时、不会用工具**四大问题。本阶段覆盖工程师日常最常碰的场景:推理优化、RAG、Agent。这也是很多公司「大模型应用工程师」岗的核心技能。

## 学习目标

- [ ] 理解 KV Cache 为什么能加速自回归生成
- [ ] 知道量化(INT8/INT4/AWQ/GPTQ)的原理与权衡
- [ ] 能用 LangChain / LlamaIndex 搭一个 RAG
- [ ] 理解向量检索(embedding + 相似度)的原理
- [ ] 说清 Agent 的 ReAct / Function Calling 机制

## 本阶段章节

- [推理优化:KV Cache / 量化](01-inference.md)
- [RAG 检索增强](02-rag.md)
- [Agent 与工具调用](03-agent.md)

## 🎯 面试高频考点(本阶段)

| 考点 | 难度 | 说明 |
|------|------|------|
| KV Cache 原理 | ★★★ | 推理优化必问 |
| 为什么量化不掉点太多 | ★★ | INT4/AWQ |
| RAG 全流程 | ★★★ | chunk + embed + retrieve + rerank |
| 向量数据库选型 | ★★ | FAISS / Milvus / Chroma |
| Agent ReAct 循环 | ★★ | Thought-Action-Observation |
| 幻觉怎么缓解 | ★★ | RAG + 对齐 + 解码策略 |

## 📂 简历项目建议

- **搭一个文档问答 RAG**(向量库 + rerank),在私有 PDF 上跑通
- **用 vLLM 部署开源模型**,对比原生 transformers 的吞吐
- **做一个能调用搜索引擎的 Agent**,展示多步推理

---

> ⏳ 正文编写中。本导览页已就绪,子页面内容将逐步填充。
