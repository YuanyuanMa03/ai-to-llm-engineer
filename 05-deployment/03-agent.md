# 03 · Agent:让模型学会用工具

> 一句话:**学完这节,你能回答「Agent 和普通 LLM 调用差在哪、ReAct 循环是什么、Function Calling 怎么让模型调用函数」这个面试题。**

## 🤔 课前小测
先别急着学,花 30 秒回答下面 3 题(答案在文末):
1. 一个普通 LLM 调用(直接 `chat(question)`)和一个 Agent,在「能做什么」上最本质的差别是什么?
2. ReAct 的循环是哪三步的反复?为什么需要循环,而不是一次推理完?
3. OpenAI 的 Function Calling,模型输出的到底是"函数的执行结果",还是"要调用哪个函数 + 参数"?

---

## 这节解决什么问题

LLM 训完是个"嘴",只会输出 token。但真实任务要的是**动作**:算一道数学题得算对、查今天天气得联网、操作数据库得执行 SQL。纯 prompt 让模型"假装"算,经常算错(它的算术是模式匹配,不是真算);让它"假装"联网,它编一个日期、编一个股价,就是幻觉的来源。

**Agent = LLM + 工具 + 循环**。它把模型从"输出文本"升级成"决定调用哪个工具 → 执行 → 看结果 → 继续推理",闭环直到任务完成。这是 2024 年后大模型应用最热的方向,也是「大模型应用工程师」面试的新晋高频区。这节讲清 Agent 的三种主流范式(ReAct、Function Calling、Multi-Agent)、怎么写一个最小 Agent、以及它在工程上为什么"看起来很美,落地很坑"。

## 核心概念 ★

### 1. Agent = 感知 → 规划 → 行动 → 观察 的闭环

普通 LLM 调用是一次性的:`输入 → 输出`,结束。Agent 是**带循环**的:

```
        ┌──── Thought(思考下一步做什么) ────┐
        │                                       │
        ▼                                       │
   Action(决定调用哪个工具 + 参数) ──► Observation(工具返回的结果)
        │                                       │
        └──────────── 循环直到完成 ◄────────────┘
```

四个动作反复:**感知**(读输入 / 观察结果)→ **规划**(决定下一步)→ **行动**(调工具)→ **观察**(看返回)。能跳出循环,通常靠模型自己输出 `Finish[最终答案]`,或设置最大步数兜底。

### 2. ReAct:把"推理"和"行动"交错进行

ReAct(Yao et al., 2022)是 Agent 最经典的范式。核心思路:**让模型在每一步先写出"我为什么这么做"(Thought),再给出动作(Action),然后看结果(Observation),再进入下一轮 Thought**。把推理链显式化,模型不会瞎调工具。

一轮的标准格式:

```
Thought: 我需要先查一下今天的股价。
Action: search_stock[Apple]
Observation: AAPL 今日收盘 $189.5
Thought: 现在我可以回答用户了。
Action: Finish[Apple 今天收盘价是 $189.5]
```

伪代码描述这个循环:

```python
def react_loop(question, tools, llm, max_steps=5):
    history = f"Question: {question}\n"
    for step in range(max_steps):
        output = llm(history)                       # 模型产出 Thought + Action
        action = parse_action(output)                # 解析出要调哪个工具
        if action.name == "Finish":
            return action.args                       # 完成,返回最终答案
        obs = tools[action.name](*action.args)       # 执行工具
        history += f"Thought: {output}\nObservation: {obs}\n"
    return "步数用尽,未能完成"
```

ReAct 的**关键设计**:Thought 是显式写出来的(写进 prompt 上下文),不是隐式的。这让模型在多步推理里**看得见自己之前在想什么**,减少"中间忘了目标"的偏移。

### 3. Function Calling:OpenAI 方式的工具调用

Function Calling 是另一种范式:**把工具描述成 JSON Schema 喂给模型,模型直接输出"要调用哪个函数 + 参数"的结构化 JSON**,由应用层去执行。

工具定义(JSON Schema):

```json
{
  "name": "get_weather",
  "description": "查询某城市当前天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "城市名"}
    },
    "required": ["city"]
  }
}
```

模型输出(注意:**它不执行,只决定**):

```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```

应用层拿这个 JSON 调真正的 `get_weather("北京")` 函数,把结果塞回 messages,再让模型继续。

### 4. ReAct vs Function Calling:推理显式与否

| 维度 | ReAct | Function Calling |
|---|---|---|
| 推理过程 | **显式**(Thought 写在文本里,可读可调) | **隐式**(模型内部决定,不暴露推理) |
| 工具描述 | 自然语言写在 prompt 里 | JSON Schema,模型专门训练过 |
| 解析方式 | 正则 / 文本解析(脆弱) | 模型直接输出 JSON(鲁棒) |
| 通用性 | 任何模型都能用(prompt 工程) | 需要模型支持 Function Calling |
| 适合 | 开源模型、需要可观察推理链 | GPT-4 / Claude / Qwen 等支持 FC 的模型 |

> ✅ **思考一下**:Function Calling 看起来比 ReAct"先进",为什么 ReAct 还活着?提示:① 开源小模型没专门训练过 FC,FC JSON 经常编不对;② 有些任务需要把 Thought 暴露给开发者审计(金融、医疗),FC 的隐式推理没法审计。两者是互补,不是替代。

### 5. Multi-Agent:多个角色协作

单个 Agent 干复杂任务容易跑偏。Multi-Agent 把任务拆给**多个有不同角色 / 工具的 Agent**,彼此协作。经典模式:

- **Planner + Worker**:一个 Agent 拆任务,一群 Agent 各干一块,最后汇总。
- **辩论**:多个 Agent 持不同观点互怼,提升事实准确性。
- **Critic / Reviewer**:一个 Agent 输出,另一个 Agent 审查、打回重做。

框架代表:AutoGen(微软,对话式多 Agent)、CrewAI(角色化分工,API 简洁)、MetaGPT(模拟软件团队)。

### 6. 规划策略:任务分解 + 反思

复杂任务一步做不出来,Agent 需要**规划**:

- **任务分解(Plan-and-Solve)**:先把大任务拆成子任务清单,逐个执行。例:"写一份市场报告" → [查数据、列大纲、写各章节、汇总]。
- **反思(Reflection)**:每步或每轮结束后,让 Agent 自己评估"做得怎么样、要不要改"。代表:Reflexion 论文——Agent 失败后写"经验教训"塞进记忆,下次避免。

> ⚠️ 反思是把双刃剑。它能救一些错误,但也可能让 Agent **陷入"自我否定循环"**——一直觉得"不够好",改了又改,永远不 Finish。所以工业 Agent 几乎都设最大步数硬截断。

### 7. 常见框架

| 框架 | 定位 | 特点 |
|---|---|---|
| **LangChain** | 全家桶 | 生态全、抽象重、学习曲线陡 |
| **LangGraph** | LangChain 出的 Agent 编排 | 把 Agent 建模成状态图,可控性强 |
| **LlamaIndex** | 偏 RAG,也支持 Agent | RAG 抽象清晰 |
| **AutoGen** | 微软,多 Agent 对话 | 对话式协作,研究友好 |
| **CrewAI** | 角色化多 Agent | API 简洁,业务上手快 |
| **OpenAI Assistants / Agents SDK** | 官方托管 | 工程最省心,但锁定生态 |

---

## 为什么这样设计

### 普通 LLM 调用 vs Agent

| 维度 | 普通 LLM(`llm.chat(q)`) | Agent |
|---|---|---|
| 能执行外部动作吗 | 不能,只会输出文本 | 能,通过工具调真实函数 / API |
| 能获取实时信息吗 | 不能,知识截止训练日 | 能,工具可联网 / 查库 |
| 能多步推理吗 | 一次出答案,长了就忘 | 循环 + 记忆,可分步 |
| 可靠性 | 相对高(prompt 稳定) | **更低**(循环、工具失败、跑偏) |
| 成本 | 一次推理 | 多次推理 + 工具调用,贵 N 倍 |

最后一行是关键:**Agent 是用"可靠性"和"成本"换"能力"**。能直接 prompt 解决的,别上 Agent;Agent 是"任务确实需要多步 + 外部工具"时的最后选择。

### ReAct vs Function Calling(再强调)

| 维度 | ReAct | Function Calling |
|---|---|---|
| 工具定义 | prompt 里用自然语言描述 | JSON Schema,结构化 |
| 输出可靠性 | 模型可能不按格式输出 → 解析失败 | 模型专门训练,JSON 鲁棒 |
| 调试 | Thought 全可见,好排查 | 推理隐式,出问题难定位 |
| 模型要求 | 任意 LLM | 需支持 FC(GPT-4 / Claude / Qwen / GLM 等) |
| 工程实践 | 教学原型、可观察性强的场景 | 生产 Agent 主流 |

## 代码:最小实现

下面用纯 Python 写一个**极简 ReAct Agent**,定义两个工具(计算器、模拟搜索),跑 Thought-Action-Observation 循环。**不依赖任何框架**,看清 Agent 的本质就是"循环 + 解析 + 调工具"。

```python
import re

# === 1. 定义两个工具 ===
def calculator(expr):
    """安全地计算一个数学表达式(这里仅演示,生产要用 ast 限定)"""
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败: {e}"

def search(query):
    """模拟一个搜索引擎(真实场景接 SerpAPI / Bing)"""
    fake_db = {"巴黎人口": "巴黎市区人口约 210 万",
               "法国首都": "法国首都是巴黎"}
    for k, v in fake_db.items():
        if k in query:
            return v
    return "未找到相关结果"

tools = {"calculator": calculator, "search": search}

# === 2. 模拟 LLM(真实场景调 OpenAI / 开源模型) ===
# 这里用规则模拟,生产换成 llm(prompt) -> str
def fake_llm(prompt):
    if "首都" in prompt and "人口" not in prompt:
        return 'Thought: 我需要查法国首都是哪。\nAction: search[法国首都]'
    if "人口" in prompt:
        return 'Thought: 我需要查巴黎人口。\nAction: search[巴黎人口]'
    if "×" in prompt or "*" in prompt:
        match = re.search(r'(calc|计算)\s*[:：]?\s*([\d\+\-\*\/\s\(\)]+)', prompt)
        expr = match.group(2).strip() if match else "2100000 * 1.2"
        return f'Thought: 我需要算一下。\nAction: calculator[{expr}]'
    return 'Thought: 信息够了,可以回答。\nAction: Finish[完成]'
```

```python
# === 3. ReAct 主循环 ===
def react_agent(question, tools, llm, max_steps=5):
    history = f"Question: {question}\n"
    for step in range(max_steps):
        print(f"\n--- 第 {step+1} 步 ---")
        output = llm(history)                       # 模型产出 Thought + Action
        print(output)
        # 解析 Action:工具名[参数]
        m = re.search(r'Action:\s*(\w+)\[(.*?)\]', output, re.S)
        if not m:
            return "解析失败,Agent 输出格式不对"
        tool_name, args = m.group(1), m.group(2).strip()
        if tool_name == "Finish":                   # 终止条件
            return args
        if tool_name not in tools:
            obs = f"工具 {tool_name} 不存在"
        else:
            obs = tools[tool_name](args)            # 执行工具
        print(f"Observation: {obs}")
        history += f"{output}\nObservation: {obs}\n"
    return "达到最大步数,Agent 未完成"

# === 4. 跑一个任务 ===
answer = react_agent("法国的首都多少人口?算一下它乘以 1.2 是多少。",
                     tools, fake_llm, max_steps=5)
print(f"\n最终答案: {answer}")
```

跑起来你会看到 Agent 走完 `search → search → calculator → Finish` 完整链路,**这就是 Agent 的本质——LLM 决策 + 工具执行 + 循环**。生产里把 `fake_llm` 换成真模型、把工具换成真 API,就是可用 Agent。

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/) 在线试跑。试着加一个新工具(比如 `weather`),让 Agent 回答"巴黎今天天气如何"。

## ⚠️ 易错点 / 面试陷阱

> **Agent 容易陷入死循环。** 模型可能反复调同一个工具、或者在两步之间反复横跳,永远不输出 Finish。**必须设最大步数硬截断**,再加"重复动作检测"(连续两次相同 Action 就强制终止或换策略)。工业 Agent 没有这个保护,迟早把 token 烧光。

> **工具描述不清,模型会乱调。** 工具的 `description` 和参数说明,是模型唯一的"使用手册"。描述含糊,模型会调错参数(把字符串塞进数字参数)、调错工具(用 calculator 去算天气)。**Function Calling 里 JSON Schema 的 description 字段,比代码本身还重要**——这是 prompt 工程的一部分。

> **Function Calling 的 JSON 解析必须鲁棒。** 模型偶尔会输出带 Markdown 代码块的 JSON(```json ... ```)、或多余文字、或截断的 JSON。生产代码必须 try-except + 修复(去 markdown fence、补全闭合括号)、解析失败要有 fallback(让模型重试或退回纯文本回答)。

> **Agent 的可靠性低于直接 prompt,不要为了"看起来高级"就上 Agent。** Agent 多了循环和工具,每一步都可能失败。能用 RAG 解决的别上 Agent,能用单次 prompt 解决的别上 RAG。**复杂度按需加**,这是工程纪律。

> **Multi-Agent 不是银弹,通信成本极高。** 多个 Agent 协作的 token 消耗是单 Agent 的数倍,而且 Agent 之间可能"互相礼貌地推卸任务"。Multi-Agent 在"任务能清晰拆解、子任务相对独立"时才有收益,否则不如一个强 Agent + 好工具。

## 🎯 面试会怎么考

- **八股题**:什么是 Agent?它和普通 LLM 调用差在哪?ReAct 循环是哪几步?Function Calling 的原理是什么?Agent 和 RAG 怎么结合(提示:RAG 本身可以包装成一个工具,Agent 决定何时检索)?
- **手撕题**:写一个简单的工具调用循环:给定工具字典和 LLM 输出格式(`Action: tool_name[args]`),写解析 + 执行 + 终止判断;或写一个 Function Calling 的 JSON 解析与容错函数。
- **深挖题**:怎么避免 Agent 死循环?Multi-Agent 怎么协作、通信协议怎么定?Agent 怎么解决"工具调用失败后自我纠正"(反思机制)?给一个具体业务(如自动客服),你怎么设计工具集和 prompt?

## 📂 简历可写的项目

**项目名:能联网搜索 + 数学计算的 ReAct Agent**

- **描述**:基于开源模型(Qwen / GLM / Llama)实现一个 ReAct Agent,工具集包括 SerpAPI 联网搜索、Python 代码解释器(沙箱)、计算器;能解决"查最新数据 + 算结果"的复合问题(例:"今天上证指数收盘多少?算一下它比昨天涨了百分之几")。带最大步数保护、重复动作检测、Action 解析容错。
- **技术栈**:LangChain / LangGraph(或纯 Python 手写)、OpenAI / 开源 LLM、SerpAPI / Tavily、Python `subprocess` 沙箱。
- **加分点**:
  - 实现**反思机制**:工具调用失败时,让 Agent 自己写"失败原因 + 下一步该怎么做"塞进上下文。
  - 在 30 个多步任务上做评测,记录成功率、平均步数、平均 token,对比"普通 LLM 直接答"的差距。
  - 用 LangGraph 把流程画成状态图,可视化整个决策路径。

## 🚀 挑战

**给上面的极简 Agent 加一个"反思"步骤**:当工具调用失败(比如 `calculator` 抛异常,或 `search` 返回"未找到"),不要直接把错误塞回循环,而是让 Agent 先**输出一段反思**——

```
Reflection: 上次搜索 "巴黎人口" 没找到,可能关键词太具体。
            下次试试 "巴黎 市区 人口 数量"。
```

把这段反思塞进 history,再进入下一轮。**对比加反思前后,Agent 在"刁钻查询"上的成功率**。

进阶:把 `fake_llm` 换成真模型(GPT-4o-mini / Qwen),给 Agent 加一个 `python_interpreter` 工具(用 `exec` + 受限 `__builtins__`),让它能跑真代码,解决"算一个复杂公式"这种文本计算会错的题。

## 🔗 延伸阅读

1. 📄 [ReAct: Synergizing Reasoning and Acting in Language Models(Yao et al., 2022)](https://arxiv.org/abs/2210.03629) —— ReAct 的原始论文,Agent 范式的奠基之作。论文里的 prompt 模板今天仍是 ReAct 实现的事实标准。
2. 📄 [Reflexion: Language Agents with Verbal Reinforcement Learning(Shinn et al., 2023)](https://arxiv.org/abs/2303.11366) —— 把"反思"做成 Agent 的标配,失败后用自然语言写经验教训塞进记忆,显著提升多步任务成功率。
3. 💻 [LangGraph 官方教程](https://langchain-ai.github.io/langgraph/) —— 把 Agent 建模成显式的状态图(节点 + 边),是目前工程上做可控 Agent 的最佳实践;比早期 LangChain 的 AgentExecutor 透明得多,适合做生产 Agent。

---

## ✅ 课后小测(答案)

**课前小测答案:**
1. **最本质的差别是"能否执行外部动作 + 能否多步循环"。** 普通 LLM 是一次性的输入→输出,只产生文本;Agent 能调真实工具(API、计算器、数据库)、能观察结果、能循环多步,从而能完成"联网查最新数据""算精确结果""改数据库"这类 LLM 单次回答做不到的任务。
2. **ReAct 的循环是 Thought(思考)→ Action(行动)→ Observation(观察)三步反复。** 需要循环是因为很多任务一步做不完——查天气要先调搜索工具,看到结果再决定要不要再查、最后才整合答案。循环让模型能"边做边想",而不是一次性瞎猜。
3. **Function Calling 模型输出的不是执行结果,而是"要调用哪个函数 + 参数"的 JSON**(`{"name": "get_weather", "arguments": {"city": "北京"}}`)。真正执行函数的是应用层(开发者代码),模型只负责决策。模型不能直接执行——它运行在沙箱里,没有外部访问权。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——

> 用户问"2024 年苹果公司的营收是多少?把它换算成人民币(按当前汇率)"。请你说出一个 Agent 完成这个任务的完整流程(用 ReAct 或 Function Calling 都行),包括每一步调什么工具、可能在哪里失败、怎么兜底。
>
> **答(ReAct 版)**:
> ① Thought:需要先查 2024 年苹果营收 → Action: `search["Apple 2024 revenue"]` → Observation: `$391B`。
> ② Thought:需要当前美元人民币汇率 → Action: `search["USD to CNY exchange rate today"]` → Observation: `7.2`。
> ③ Thought:需要算 391 × 7.2 → Action: `calculator["391 * 7.2"]` → Observation: `2815.2`。
> ④ Thought: 信息齐全 → Action: `Finish["约 2815.2 亿元人民币"]`。
>
> 失败点 + 兜底:① 搜索结果可能给的是季度而非全年(描述不清);② 汇率网站数据延迟或冲突(多源校验);③ 算式里单位搞错(亿美元 vs 美元 → 差 100 倍);④ Agent 卡在第 ② 步反复搜索(设最大步数 + 重复动作检测)。Agent 的工程难点不在"跑通",而在这些边界 case 的兜底——这也是面试官最爱深挖的地方。
