<div align="center">

# AI 手帐 · LLM Engineer Quest

### 把大模型学成主线任务，把 Python 跑成打怪现场。

一座面向中文学习者与求职者的「边学、边跑、边打怪」交互式实验场。<br>
从数学地基、Transformer、预训练与对齐，一路打到 RAG、Agent、部署和 MCP；代码不只负责排版，还要当场交作业。

[![在线体验](https://img.shields.io/badge/在线体验-打开手帐-EF4A43?style=for-the-badge&logo=githubpages&logoColor=white)](https://yuanyuanma03.github.io/ai-to-llm-engineer/)
[![面试试炼](https://img.shields.io/badge/面试试炼-31_道题-12305A?style=for-the-badge)](https://yuanyuanma03.github.io/ai-to-llm-engineer/quiz/)
[![课程关卡](https://img.shields.io/badge/课程关卡-16_关-F2C84B?style=for-the-badge&labelColor=12305A)](#学习路线)
[![浏览器实验](https://img.shields.io/badge/Python-Pyodide_实验台-16847D?style=for-the-badge&logo=python&logoColor=white)](#代码不再负责看起来能跑)
[![Language](https://img.shields.io/badge/Language-中文-16847D?style=for-the-badge)](#)

**[开始学习](https://yuanyuanma03.github.io/ai-to-llm-engineer/)** · **[打开面试题库](https://yuanyuanma03.github.io/ai-to-llm-engineer/quiz/)** · **[练习手撕代码](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/hand-coding)**

</div>

<a href="https://yuanyuanma03.github.io/ai-to-llm-engineer/">
  <img src="output/playwright/readme-home.webp" alt="AI 手帐首页：学习进度、今日任务与六阶段冒险地图" width="100%">
</a>

> 不是资源链接农场，也不是“看完等于学会”的电子安慰剂。这里把知识拆成关卡，把练习变成任务，把每次真正掌握变成可以盖章的进度。

## 当前版本，一眼看懂

| 模块 | 已上线内容 | 学习者拿到什么 |
|---|---:|---|
| **课程地图** | 6 个阶段 · 16 个关卡 | 从数学、深度学习到 Agent / MCP 的连续路线 |
| **浏览器实验台** | Python · NumPy · Matplotlib · 跨代码格运行 | 编辑后立即看到文本、异常和图表反馈 |
| **手撕训练** | 10 组可运行代码练习 | 把“我大概会”升级成“我可以现场写” |
| **面试试炼** | 31 道分阶段题卡 | 随机抽题、难度筛选与掌握度自评 |
| **游戏化进度** | 任务、XP、实验星、等级、连续学习天数 | 每次行动都有反馈，收藏夹终于不能代练 |
| **运行方式** | 纯静态站点 · GitHub Pages | 无账号、无后端，打开网页即可学习 |

```text
读懂直觉 → 改一行代码 → 浏览器运行 → 查看输出 / 图表 / 报错
    → 自动勾选“跑通代码” → +20 XP / 实验星 → 完成本关盖章
```

## 为什么是一本「AI 手帐」

学习大模型最容易出现一种幻觉：收藏了 47 篇文章，于是感觉自己已经懂了 47 篇。AI 手帐想做的，是把这种“知识囤积”改造成一条有输入、有动作、有验收的工程路线。

| 你会得到 | 网页里如何实现 |
|---|---|
| **一条完整路线** | 6 个阶段、16 个课程关卡，从数学基础到 MCP |
| **真实学习反馈** | 任务清单、XP、实验星、等级、连续学习天数与通关印章 |
| **能跑的工程练习** | 浏览器内 Python 实验台、即时输出/报错/图表与 10 道手撕代码题 |
| **能讲清的面试能力** | 高频追问、易错点、31 道随机题卡与掌握度自评 |
| **能写进简历的证据** | 每阶段都给出项目方向、交付物与验收口径 |

## 网页里有什么

### 课程不是文章列表，而是一张可推进的任务地图

- 首页会根据本地进度推荐下一关，不用每天重新思考“我该学什么”。
- 每关必须完成「读懂直觉 / 跑通代码 / 回答面试题」三项任务才能盖章。
- XP、等级、通关状态和连续学习天数保存在浏览器本地，无需注册账号。
- 所有课程都有上一关 / 下一关导航，学到一半不会掉进链接迷宫。

<a href="https://yuanyuanma03.github.io/ai-to-llm-engineer/#/01-fundamentals/01-math">
  <img src="output/playwright/readme-lesson.webp" alt="AI 手帐课程页：教程正文、学习路线与本关任务卡" width="100%">
</a>

### 代码不再负责“看起来能跑”

每个 Python 代码块都是一格真正的迷你实验室：可以直接编辑、重置和运行。标准库、NumPy、Matplotlib 等代码在浏览器的独立线程中执行，页面不会因为一段计算当场失去意识；输出、异常和图表会原地回贴到手帐。

<a href="https://yuanyuanma03.github.io/ai-to-llm-engineer/#/01-fundamentals/01-math">
  <img src="output/playwright/readme-code-lab.webp" alt="AI 手帐浏览器 Python 实验台：编辑 NumPy 代码、即时查看输出并获得实验星" width="100%">
</a>

- 点「运行到这里」会自动补跑本页上方的兼容代码块，跨格变量不会突然失忆。
- 首次运行按需加载 Python 与科学计算包；之后复跑通常只需数秒。
- 跑通后自动勾选本关「跑通代码」，首次胜利获得 `+20 XP` 和一颗实验星。
- 无限循环会触发急停；错误会变成可读的 Bug Boss 面板，而不是一堵红色天书。
- PyTorch、Transformers、MCP Server 等重型或服务端依赖会明确标记为「云端关卡」，仍可编辑和复制，不会假装浏览器什么都能炼。

#### 实验台能力边界

| 代码类型 | 网页内运行 | 反馈方式 |
|---|:---:|---|
| Python 标准库 | ✅ | 标准输出、表达式结果、可读异常 |
| NumPy / 可由 Pyodide 加载的科学计算包 | ✅ | 首次按需加载，后续复用运行时 |
| Matplotlib | ✅ | 自动捕获当前图表并贴回代码格下方 |
| 同一页的前置兼容代码格 | ✅ | 「运行到这里」按顺序补跑，共享变量与函数 |
| `input()` 交互 | 暂不支持 | 引导改成变量后运行 |
| PyTorch / Transformers / vLLM / MCP Server | ☁️ 云端关卡 | 保留编辑、复制与本地 / Notebook 运行提示 |

每次执行都有两道保险：运行阶段超过 20 秒会自动急停，运行时或依赖加载超过 120 秒会提示检查网络。手动点击「停止」会直接终止 Worker，下一次运行会得到一间干净的新实验室。

### 每一关都有明确的学习闭环

```text
课前摸底  →  直觉解释  →  公式与最小实现  →  易错点
    →  面试追问  →  简历项目  →  挑战任务  →  课后验收
```

内容参考 Microsoft AI-For-Beginners 的教学闭环，并围绕大模型工程师岗位重新组织：不只告诉你“这是什么”，还要回答“为什么这样设计、怎么跑通、面试怎么问、最终留下什么证据”。

## 学习路线

| 阶段 | 关卡 | 最终战利品 |
|---|---|---|
| **01 · 基础补给站** | [数学基础](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/01-fundamentals/01-math) · [Python 与数据栈](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/01-fundamentals/02-python) · [机器学习范式](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/01-fundamentals/03-ml-basics) | 看懂公式，写出向量化实现，讲清训练与评估 |
| **02 · Transformer 山谷** | [反向传播](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/02-deep-learning/01-nn) · [CNN / RNN](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/02-deep-learning/02-cnn-rnn) · [Transformer](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/02-deep-learning/03-transformer) | 手写 Attention，解释计算图与架构取舍 |
| **03 · 预训练矿场** | [Tokenizer](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/03-pretraining/01-tokenizer) · [Decoder-only / MoE](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/03-pretraining/02-architecture) · [Scaling Laws](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/03-pretraining/03-scaling-laws) | 拆开现代 LLM 的发动机舱，算清数据与算力配比 |
| **04 · 对齐试炼场** | [SFT](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/04-finetuning/01-sft) · [LoRA / QLoRA](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/04-finetuning/02-peft) · [RLHF / DPO](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/04-finetuning/03-alignment) | 用更少显存完成微调，解释偏好对齐的工程逻辑 |
| **05 · 部署发射台** | [推理优化](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/05-deployment/01-inference) · [RAG](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/05-deployment/02-rag) · [Agent](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/05-deployment/03-agent) | 构建能检索、能调用工具、能被评估的 LLM 应用 |
| **06 · 前沿观测站** | [MCP 工具接入协议](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/06-frontier/01-mcp) | 给 Agent 的工具箱装上统一接口 |

## 三种打开方式

### 完全小白

从 [01-1 数学基础](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/01-fundamentals/01-math) 开始，按地图推进。别急着跳到 Agent——地基不会因为你很有热情就自动出现。

### 已有机器学习基础

从 [Transformer 结构详解](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/02-deep-learning/03-transformer) 开始，用每关的面试追问快速查漏补缺。

### 正在准备面试

直接进入 [面试试炼场](https://yuanyuanma03.github.io/ai-to-llm-engineer/quiz/) 和 [手撕代码题合集](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/hand-coding)，先答题，再回到对应关卡补洞。

## 交互实验场是怎么工作的

```mermaid
flowchart LR
    A["课程 Markdown"] --> B["Docsify 渲染"]
    B --> C["代码格识别与编辑器"]
    C --> D["Web Worker"]
    D --> E["Pyodide / Python"]
    E --> F["stdout / traceback / Matplotlib"]
    F --> G["原地反馈"]
    G --> H["任务进度 / XP / 实验星"]
    H --> I["localStorage"]
```

- [`assets/code-lab-core.js`](assets/code-lab-core.js)：判断代码能否在浏览器运行、生成稳定实验 ID、压缩 Python 异常。
- [`assets/code-lab.js`](assets/code-lab.js)：把课程代码块升级成编辑器，管理运行、急停、重置、复制和结果展示。
- [`assets/python-worker.mjs`](assets/python-worker.mjs)：在独立 Worker 中加载 Pyodide、按需安装包、顺序执行代码格并捕获图表。
- [`assets/app.js`](assets/app.js)：连接课程任务、通关状态、XP、实验星和首页玩家信息。
- [`tests/code-lab-core.test.cjs`](tests/code-lab-core.test.cjs)：覆盖代码分类、稳定 ID 与错误清理等核心行为。

> **边界说明：** Worker 能避免长任务直接卡住页面，但它不是用来运行不可信代码的强安全沙箱。课程实验代码来自本仓库，学习者也应只运行自己理解或信任的代码。

## 本地运行

项目基于 Docsify，无需安装依赖或执行构建：

```bash
git clone https://github.com/YuanyuanMa03/ai-to-llm-engineer.git
cd ai-to-llm-engineer
python3 -m http.server 4173
```

浏览器访问 `http://127.0.0.1:4173/`。学习进度与实验星仅写入浏览器 `localStorage`，代码在本机浏览器的 Web Worker 中执行，不会上传到本项目的服务器。首次运行需要联网从 CDN 加载 Pyodide 与所需科学计算包。

维护者可以额外运行零依赖检查：

```bash
npm run check
npm test
```

当前质量门槛：4 个 JavaScript 文件通过语法检查，5 项实验台核心测试必须全部通过。仓库没有前端构建产物，Markdown、CSS 与原生 JavaScript 的修改刷新页面即可看到。

## 项目结构

```text
ai-to-llm-engineer/
├── 01-fundamentals/ ... 06-frontier/  # 6 阶段课程正文
├── assets/                            # 手帐主题、游戏进度、实验台与 Worker
├── quiz/                              # 31 道面试试炼题
├── tests/                             # 浏览器实验台核心测试
├── output/playwright/                 # README 使用的真实网页截图
├── hand-coding.md                     # 10 组手撕代码练习
├── home.md                            # 冒险地图首页
├── TEMPLATE.md                        # 新课程写作模板
└── index.html                         # Docsify 入口
```

## 内容与技术

- **教学结构**：课前摸底、直觉解释、可运行实验、课后测验与求职验收。
- **前端实现**：Docsify、原生 JavaScript、CSS、KaTeX、Prism、[Pyodide](https://pyodide.org/) Web Worker；零框架构建链，避免“先学课程，还得先修 webpack”。
- **参考项目**：[Microsoft AI-For-Beginners](https://github.com/microsoft/AI-For-Beginners)；部分手绘知识图来自该项目，并保留来源说明。
- **原创范围**：围绕大模型工程师求职重新规划的中文课程、工程实践、面试追问、手撕代码与游戏化学习系统，并非原项目的逐章翻译。

## 一起把它写得更好

- 发现知识或代码错误：欢迎提交 Issue，或通过课程页底部直接定位源文件。
- 想补充课程：请沿用 [写作模板](TEMPLATE.md)，保证有输入、实验与验收。
- 遇到新的面试题：欢迎在 [Discussions](https://github.com/YuanyuanMa03/ai-to-llm-engineer/discussions) 投稿。

<div align="center">

如果这本手帐帮你少走了一点弯路，欢迎点一颗 Star。<br>
收藏不能替你学习，但 Star 可以提醒作者继续更新。⭐

**[现在开始第 1 关 →](https://yuanyuanma03.github.io/ai-to-llm-engineer/#/01-fundamentals/01-math)**

</div>
