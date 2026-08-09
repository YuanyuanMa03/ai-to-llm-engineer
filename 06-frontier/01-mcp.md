# 01 · MCP:让 AI 标准化接入工具的协议

> 一句话:**学完这节,你能回答「MCP 解决了什么问题、Host/Client/Server 三角色各自干什么、和 Function Calling 是什么关系」这个面试题。**

## 🤔 课前小测

先别急着学,花 30 秒回答下面 3 题(答案在文末):

1. 在 MCP 出现之前,要让 Claude、GPT、Gemini 三个模型各自接入 GitHub、Notion、Slack 三个工具,大概要写几份集成代码?为什么这是个"组合爆炸"问题?
2. MCP 架构里有 Host、Client、Server 三个角色。一个 Host(比如 Claude Desktop)同时连了 3 个 MCP Server,它内部会有几个 Client?
3. 有人说"MCP 就是要取代 Function Calling",这句话对吗?

---

## 这节解决什么问题

大模型本身只会"说话",不会"动手"——它不能查你的数据库、不能读你的日历、不能调你的内部 API。要让 AI 真正干活(Agent),必须给它接工具。

但 2024 年之前,接工具是一件**苦活**:OpenAI 一套 Function Calling 格式,Anthropic 一套 tool_use 格式,Google 又一套;GitHub、Notion、Slack 每个工具还得各自封装。结果是 **N 个模型 × M 个工具 = N×M 份适配代码**,谁都没动力写第 N×M+1 份。

2024 年 11 月,Anthropic 开源了 **MCP(Model Context Protocol)**,用一个统一协议把这件事变成 **N+M**:工具方只写一份 MCP Server,模型方只实现一个 MCP Client,双方按标准握手就能用。到 2025-2026 年,Claude、ChatGPT、Cursor、VS Code、ZCode 等主流宿主全线支持,MCP 已成 Agent 接入工具的**事实标准**。这一节讲清它的架构、核心概念、最小实现,以及它和 Function Calling 到底什么关系——这是当下 Agent / AI 工具链岗位的最高频考点。

## 核心概念 ★

### 1. MCP 是什么:AI 世界的 "USB-C 接口"

MCP 是一个**开源协议**,规定 AI 应用如何与外部系统(数据源、工具、服务)通信。官方有个很贴切的比喻:**MCP 之于 AI 应用,就像 USB-C 之于电子设备**——一个标准接口,插什么都通用。

它的底层是 **JSON-RPC 2.0**(一种轻量远程调用协议),规定了消息格式、发现机制、能力协商。任何遵循这个协议的 Server 和 Client 都能互通,不绑定具体厂商。

### 2. 三角色架构:Host / Client / Server

这是理解 MCP 的核心。官方文档明确定义三个参与者:

| 角色 | 是什么 | 例子 |
|------|--------|------|
| **Host(宿主)** | 用户直接交互的 AI 应用,负责管理多个连接 | Claude Desktop、Cursor、VS Code Copilot |
| **Client(客户端)** | Host 内部的连接器,与一个 Server 维持 **1:1 专用连接** | Host 每连一个 Server 就实例化一个 Client |
| **Server(服务端)** | 暴露工具/数据/提示模板的程序,可本地可远程 | filesystem server、GitHub server、Sentry server |

关键关系:**1 个 Host 内部有多个 Client,每个 Client 1:1 连一个 Server**。Host 是"总管家",Client 是"专线",Server 是"工具提供方"。架构图:

```
┌──────────────────────── Host (AI 应用,如 Claude Desktop) ────────────────────────┐
│                                                                                    │
│   Client 1 ──── 1:1 专用连接 ────► Server A (本地, stdio, 如 filesystem)            │
│   Client 2 ──── 1:1 专用连接 ────► Server B (本地, stdio, 如 database)             │
│   Client 3 ──── 1:1 专用连接 ────► Server C (远程, HTTP, 如 Sentry)                │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

> ✅ **思考一下**:为什么不让 Host 直接连 Server,非要中间塞一层 Client?提示:想想"隔离"和"多 Server 并发管理"——每个 Client 是一条独立通道,某个 Server 崩了不会拖死别的连接,Host 也不用自己维护 N 条状态机。

### 3. 三大核心能力(Primitives)

MCP Server 能向 Client 暴露三种"原语",这是它提供能力的全部方式:

| 原语 | 作用 | 控制方 | 例子 |
|------|------|--------|------|
| **Tools(工具)** | 可执行的函数,LLM 决定调用 | 模型控制(model-controlled) | `query_db`、`send_email`、`search_code` |
| **Resources(资源)** | 可读取的数据,应用决定何时读 | 应用控制(app-controlled) | 文件内容、数据库 schema、API 返回 |
| **Prompts(提示模板)** | 预定义的交互模板,用户选用 | 用户控制(user-controlled) | "总结这个仓库"、"调试这个错误" |

记忆口诀:**Tools 给模型用、Resources 给应用用、Prompts 给用户选**。三者对应不同的"控制权归属",这是 MCP 设计里很重要的一条线。

### 4. 传输层:stdio 与 Streamable HTTP

MCP 把协议分成两层:**数据层**(JSON-RPC 消息语义)和**传输层**(怎么传字节)。传输层有两种机制:

- **stdio(标准输入输出)**:本地子进程通信,Host 把 Server 当子进程拉起,通过 stdin/stdout 收发。无网络开销,适合**本地**场景(读写本地文件、查本地数据库)。注意:stdio Server 千万别用 `print()`——stdout 是协议通道,写脏数据会破坏 JSON-RPC 消息,只能用 stderr。
- **Streamable HTTP**:用 HTTP POST 传消息,可选 Server-Sent Events 做流式。适合**远程** Server(部署在云端、跨网络),支持 OAuth/Bearer Token 等标准鉴权。

早期文档里的 "SSE transport" 已升级为 "Streamable HTTP"。本地用 stdio、远程用 HTTP,是选型基本盘。

### 5. 和 Function Calling 的关系:标准化,不是取代

这是最常被问的点。一句话:**Function Calling 是"模型会不会调工具"的能力,MCP 是"工具怎么被标准化接入"的协议,二者互补**。

- **Function Calling**(OpenAI 2023 推出):模型层面,LLM 能理解工具描述、决定何时调用、生成结构化参数。它解决"模型懂不懂用工具"。
- **MCP**:协议层面,规定工具如何被发现(`tools/list`)、如何被调用(`tools/call`)、返回格式如何。它解决"工具方和应用方怎么标准化对接"。

实际流程里两者是配合的:Host 从各 MCP Server 拉到工具列表 → 转成模型能理解的 Function/Tool 描述 → 模型用 Function Calling 能力决定调哪个 → Host 把调用请求按 MCP 协议发回对应 Server → Server 执行后返回结果。MCP 没有取代 Function Calling,反而**依赖**它来驱动工具选择。

> ✅ **思考一下**:如果有个模型压根不支持 Function Calling,它还能用 MCP 吗?提示:能用 Resources/Prompts(应用/用户主动拉),但 Tools 的"模型自主决定调用"这条路径就走不通了。

### 6. 一个量化的直觉:N×M → N+M

这是 MCP 价值最直观的数学表达。设要对接 $N$ 个 AI 应用和 $M$ 个工具:

$$
\text{无 MCP 的适配数} = N \times M, \qquad \text{有 MCP 的适配数} = N + M
$$

当 $N=10$、$M=20$ 时:无 MCP 要写 $200$ 份对接,有 MCP 只需 $30$ 份(10 个 Client 实现 + 20 个 Server)。规模越大,收益越夸张。这就是"协议标准化"的杠杆效应——和当年 USB、HTTP、SQL 统一接口是同一个道理。

---

## 为什么这样设计

### 对比一:无 MCP vs 有 MCP

| 维度 | 无 MCP(各自为政) | 有 MCP(统一协议) |
|------|-------------------|-------------------|
| 对接复杂度 | $O(N \times M)$,组合爆炸 | $O(N + M)$,线性可扩 |
| 工具复用 | 换个模型要重写 | 写一次 Server,所有 Host 通用 |
| 模型切换 | 锁死在某个生态 | 换 Host 不动 Server |
| 生态 | 厂商各自闭环 | 开放生态,Server 可被社区共享 |

### 对比二:Function Calling vs MCP

| 维度 | Function Calling | MCP |
|------|------------------|-----|
| 层次 | 模型能力(模型懂不懂用工具) | 接入协议(工具怎么标准化对接) |
| 提出方 | 各模型厂商各自实现 | Anthropic 主导,开源标准 |
| 粒度 | 单次调用(JSON 参数) | 完整生命周期(发现/调用/通知) |
| 关系 | 被 MCP **依赖**(驱动工具选择) | **标准化** Function Calling 的落地 |

---

## 代码:最小实现

下面用官方 Python SDK 写一个**极简 MCP Server**:暴露一个"算加法"的工具。环境装 `mcp` 包(`uv add mcp[cli]` 或 `pip install mcp`)。基于官方 SDK 2.0+ 的 `MCPServer` 类。

### 第 1 段:写一个 MCP Server(20 行)

```python
# server.py —— 一个最小的 MCP Server
from mcp.server import MCPServer

mcp = MCPServer("calc")  # 初始化,名字叫 calc

@mcp.tool()              # 这个装饰器把函数注册成 MCP Tool
def add(a: int, b: int) -> int:
    """两数相加,返回和。

    Args:
        a: 第一个数
        b: 第二个数
    """
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")   # 以 stdio 模式运行,等 Host 连接
```

注意几个要点:`MCPServer` 类会**自动**根据函数的类型注解(`a: int`)和 docstring 生成工具的输入 schema,不用手写 JSON Schema。stdio 模式下**别用 `print()`**,要打日志只能写 stderr(`logging` 模块默认走 stderr,安全)。

### 第 2 段:配置 Host 接入(Claude Desktop 配置文件)

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "calc": {
      "command": "python",
      "args": ["/绝对路径/server.py"]
    }
  }
}
```

重启 Claude Desktop 后,它会拉起这个子进程,通过 stdio 建立连接,`add` 工具就出现在对话里了。问它"算一下 17 加 25",Claude 会自动调用 `add(17, 25)` 得到 42。

### 第 3 段:纯 Python 模拟调用流程(不依赖 SDK,看清协议本质)

如果环境装不了 `mcp` 包,下面这段纯 Python 模拟了 MCP 的核心交互——**发现工具 + 调用工具**,帮你理解协议层在干什么:

```python
import json

# 模拟一个 MCP Server 暴露的工具列表(tools/list 的响应)
tools = [
    {
        "name": "add",
        "description": "两数相加",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    }
]

# 1) 模拟 Client 发起 tools/list 请求(JSON-RPC 2.0 格式)
list_request = {
    "jsonrpc": "2.0", "id": 1, "method": "tools/list",
    "params": {}
}
print("Client → Server:", json.dumps(list_request, ensure_ascii=False))
print("Server → Client:", json.dumps({"tools": tools}, ensure_ascii=False))

# 2) 模拟模型决定调用 add,Client 发起 tools/call
call_request = {
    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
    "params": {"name": "add", "arguments": {"a": 17, "b": 25}}
}
print("Client → Server:", json.dumps(call_request, ensure_ascii=False))

# 3) Server 执行并返回结果
result = {"jsonrpc": "2.0", "id": 2, "result": {
    "content": [{"type": "text", "text": "42"}]
}}
print("Server → Client:", json.dumps(result, ensure_ascii=False))
# 输出 42 —— 这就是一次完整的 MCP 工具调用
```

这段揭示了 MCP 的本质:**就是一堆约定好格式的 JSON-RPC 消息来回传**。所有花哨的 SDK,底层都是这个。

> 💡 把代码复制到 [JupyterLite](https://jupyterlite.github.io/demo/) 在线试跑。第 3 段纯标准库,直接能跑,帮你建立对协议的直觉。

---

## ⚠️ 易错点 / 面试陷阱

> **陷阱 1:"MCP 取代 Function Calling"——错。** MCP 是接入协议,Function Calling 是模型能力,二者互补。MCP 的 Tools 调用链路恰恰依赖模型的 Function Calling 能力来决定"何时调、调哪个"。正确说法:MCP **标准化**了工具的接入方式,而 Function Calling 是这链条上的"决策引擎"。

> **陷阱 2:把 Host 和 Client 混为一谈。** Host 是 AI 应用整体(如 Claude Desktop),Client 是 Host 内部为**每个** Server 单独实例化的连接器,1:1 对应。一个 Host 连 3 个 Server,内部就有 3 个 Client 对象,各自独立。说"Client 和 Server 是一对一关系"是对的,说"Host 和 Server 一对一"就错了。

> **陷阱 3:stdio Server 里用 `print()` 调试。** 这是新手必踩的坑。stdio 传输下,stdout 是 JSON-RPC 协议通道,任何非协议字节都会**破坏消息帧**,导致 Server 直接哑火。调试只能用 stderr(Python 的 `logging` 默认走 stderr,安全;`print` 默认走 stdout,危险)。官方文档反复强调这一条。

> **陷阱 4:以为 MCP 只有一种传输方式。** 实际有 stdio(本地)和 Streamable HTTP(远程)两种。早期文档里的 "SSE transport" 已被 Streamable HTTP 取代,面试时说"SSE"不算全错(底层确有 SSE 做流式),但说"Streamable HTTP"更准确、更显前沿。

---

## 🎯 面试会怎么考

- **八股题**:MCP 的三角色(Host/Client/Server)分别是什么?Server 能暴露哪三种原语?MCP 解决了什么问题(说出 N×M → N+M)?MCP 用什么作为底层协议(JSON-RPC 2.0)?
- **手撕题**:现场写一个最小 MCP Server(用 `@mcp.tool()` 注册一个工具);或画出"用户提问 → 模型决策 → MCP 调用工具 → 返回结果"的完整时序流程。
- **深挖题**:MCP 和 Function Calling 是什么关系?为什么 stdio 模式下不能用 `print`?Streamable HTTP 相比纯 SSE 有什么改进?MCP 的安全考量(Server 能访问本地文件,如何做权限控制和沙箱)?

---

## 📂 简历可写的项目

**项目名:MCP 自定义工具集 —— 给 AI 助手接入内部能力**

**描述**:用 MCP 协议为 Claude / Cursor / VS Code 编写一组自定义工具 Server,让 AI 助手能直接查询本地 SQLite 数据库、调用公司内部 REST API、读取 Obsidian 笔记库。基于官方 Python SDK(`MCPServer` + `@mcp.tool()`),支持 stdio 本地传输,工具列表动态注册。

**技术栈**:Python、MCP SDK、JSON-RPC、SQLite、FastAPI、Claude Desktop / Cursor 配置。

**亮点**:把原本散落的脚本统一成标准协议,新员工接入新工具只需写一个 `@mcp.tool()` 函数;对比之前为每个 AI 应用单独写适配,维护成本从 N×M 降到 N+M。

---

## 🚀 挑战

给你的开发环境做一个 MCP Server,让 AI 助手能查你的**日历**或**笔记**:

1. 选一个你日常用的本地数据源(本地 Markdown 笔记目录、SQLite 数据库、iCal 文件均可)。
2. 用 `MCPServer` + `@mcp.tool()` 暴露 1~2 个工具(如 `search_notes(keyword)`、`list_events(date)`)。
3. 配置到 Claude Desktop 或 Cursor,问它"帮我找一下我记过关于 transformer 的笔记"。
4. 进阶:再加一个 **Resource**(暴露笔记的目录树)和一个 **Prompt**(预设"总结这篇笔记"的模板),体会三种原语的控制权差异。

做完这个,你不仅会用 MCP,还能在面试里拿出实物——这是"懂协议"和"只会背概念"的分水岭。

---

## 🔗 延伸阅读

- 📄 [MCP 官方文档与规范](https://modelcontextprotocol.io/introduction) —— 权威一手资料,Architecture 章节讲清三角色和原语,务必读
- 💻 [MCP 官方 Servers 仓库](https://github.com/modelcontextprotocol/servers) —— filesystem、git、github 等参考实现,读源码学最佳实践
- 🎥 [Anthropic 官方 MCP 发布博客](https://www.anthropic.com/news/model-context-protocol) —— 看 Anthropic 自己怎么讲"为什么做这个协议",理解设计动机

---

## ✅ 课后小测(答案)

**课前小测答案:**

1. **要写 3×3=9 份**(Claude×GitHub、Claude×Notion、Claude×Slack、GPT×GitHub、……)。因为每对模型-工具组合都要单独适配,这就是 $N \times M$ 的组合爆炸,规模一上来就维护不动。
2. **3 个 Client**。Host 内部每连一个 Server 就实例化一个 Client,Client 与 Server 是 1:1 专用连接。Host 是总管家,Client 是专线。
3. **不对**。MCP 是接入协议(规定工具如何被发现和调用),Function Calling 是模型能力(模型决定何时调工具)。两者互补,MCP 的 Tools 调用链路恰恰依赖 Function Calling 来驱动决策。正确说法是 MCP **标准化**了工具接入,而非取代。

**掌握自检:** 如果你能不看笔记回答下面这题,说明这节过关了 ——

> 假设你要给公司内部搭一个 AI 助手,让它能查订单数据库、读 Confluence 文档、发 Slack 消息。请用 MCP 架构描述:谁是 Host、要起几个 Server、每个 Server 暴露哪些原语(Tool/Resource/Prompt)、本地和远程 Server 分别用什么传输、整条链路上 Function Calling 在哪一步发挥作用。
