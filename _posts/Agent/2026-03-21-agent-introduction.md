---
title: "AGENT 导论"
date: 2026-03-21
description: "以 LLM 为核心的自治系统：从概念、架构到 Claude Code、OpenClaw、Codex 的技术一览。"
---

## 一、什么是 Agent

**一句话定义**：Agent 是一个以 LLM 为推理核心、能够感知环境、自主规划、调用工具并执行多步骤任务的自治系统。

与传统软件和传统 AI 的核心区别在于：

| 维度 | 传统软件 | 传统 AI（LLM Chat） | Agent |
|------|---------|-------------------|-------|
| 控制流 | 确定性，预先编码 | 单次响应 | 动态，模型自主决策 |
| 工具使用 | 固定 API 调用 | 无 | 动态选择并调用工具 |
| 状态 | 显式状态机 | 无状态 | 有状态，持久记忆 |
| 任务范围 | 单一功能 | 单轮问答 | 多步骤、跨系统任务 |
| 错误处理 | 硬编码逻辑 | 无法恢复 | 自适应重规划 |

从人机关系的视角：
- **传统工具**：人主动使用，工具被动响应
- **AI 助手（Copilot）**：人驾驶，AI 给建议
- **Agent**：人定义目标，AI 自主执行，人监督结果

> Agent 不是更聪明的聊天机器人，而是能够代理人类完成工作的软件实体。

---

## 二、Agent 的历史跃迁

### 2.1 技术汇流点

2025 年之所以成为 Agent 元年，是多项技术同时成熟的结果：

```
Reasoning Leap       Tool-Call Standards    Context Expansion
(o3, Claude 3.5+)  → (MCP, A2A Protocol) →  (200k-1M Tokens)
        ↘                   ↓                    ↙
                   Agents enter production
        ↗                   ↑                    ↖
Sandbox Runtimes      Framework Maturity     OSS Community Boom
(E2B, Docker)        (LangGraph, CrewAI)     (OpenClaw 180k ⭐)
```

### 2.2 市场数据

- Gartner 统计：Multi-agent 系统咨询量从 2024 Q1 到 2025 Q2 增长 **1,445%**
- 2026 年底预计 40% 企业应用将嵌入 AI Agent（2025 年不足 5%）
- Agent 市场规模预计从 78 亿美元增长至 2030 年的 **520 亿美元**
- OpenClaw 在 72 小时内突破 60,000 GitHub Stars，成为有史以来增长最快的开源项目之一

### 2.3 范式的转移

2025 年的核心转移可以用一句话概括：

> **AI 从「接口」变成了「基础设施」。**

语言不再只是输出的载体，而成为了通用指令接口；协调（Orchestration）、上下文（Context）、互操作性（Interoperability）成为了一等问题。

---

## 三、Agent 的核心组成

任何一个 Agent 系统，无论是 Claude Code、OpenClaw 还是 OpenAI Codex，在架构上都由五个核心组件构成：

```
┌──────────────────────────────────────────────────┐
│                   Agent Core                     │
│                                                  │
│  ┌─────────┐   ┌─────────┐   ┌─────────────┐     │
│  │  Brain  │   │ Memory  │   │  Tool Suite │     │
│  │  (LLM)  │◄──│         │   │             │     │
│  └────┬────┘   └─────────┘   └──────┬──────┘     │
│       │                             │            │
│  ┌────▼────────────────────────┐    │            │
│  │    Planner                  │◄───┘            │
│  │  ReAct / Plan-Execute /     │                 │
│  │  Chain-of-Thought           │                 │
│  └────┬────────────────────────┘                 │
│       │                                          │
│  ┌────▼────────────────────────┐                 │
│  │    Executor                 │                 │
│  │  Tool Calls / Shell / API   │                 │
│  └─────────────────────────────┘                 │
└──────────────────────────────────────────────────┘
         ↑                        ↓
   [Perceive env]           [Act on env]
 (Files/API/UI/Msg)   (Write/Create/Send/Call)
```

### 3.1 大脑（LLM）

LLM 是 Agent 的推理中枢，负责解析任务、制定计划、选择工具、理解工具结果。不同的 Agent 对底层模型有不同的假设：

- **Claude Code**：绑定 Anthropic 模型，深度优化 Claude 的推理风格
- **OpenClaw**：模型无关（Model-Agnostic），支持 Claude、GPT、Gemini、本地模型
- **Codex CLI**：绑定 OpenAI 模型，以 GPT-5.3-Codex 为核心

选模型时的核心权衡：**推理质量 × 速度 × 成本**。Plan-Execute 分离模式（规划用强模型，执行用快模型）可将成本降低高达 90%。

### 3.2 记忆（Memory）

记忆是赋予 Agent 跨会话智能的关键。通常分为三层：

| 层次 | 存储位置 | 时效 | 典型实现 |
|------|---------|------|---------|
| **短期记忆** | 上下文窗口 | 会话内 | 完整对话历史 |
| **工作记忆** | 工具结果/中间状态 | 任务内 | TODO 列表、Scratchpad |
| **长期记忆** | 文件/数据库 | 跨会话持久 | CLAUDE.md / MEMORY.md / 向量库 |

一个关键的工程洞察：**Claude Code 和 OpenClaw 均选择了 Markdown 文件而非向量数据库**作为长期记忆载体——零运维成本、可 Git 版本化、可人工审查，是「简单优先」哲学的体现。

### 3.3 工具集（Tool Suite）

工具是 Agent 的手，让它从「说」变成「做」。工具可分为四类：

```
Read Tools        Write Tools       Execute Tools     Comms Tools
──────────        ───────────       ─────────────     ───────────
• File read       • File write      • Shell cmd       • HTTP/API
• Dir search      • Code edit       • Code run        • Messaging
• Full-text grep  • DB write        • Browser auto    • Email/Cal
• Web crawl       • Form submit     • Container ops   • Ext services
```

工具定义的质量直接影响 Agent 行为：工具描述过于模糊会导致错误选择，暴露过多工具会稀释注意力。建议：**每个 Agent 实例的工具数量控制在 15 个以内**。

### 3.4 规划器（Planner）

规划器决定 Agent 「怎么想」，是将任务转化为行动序列的核心机制。三种主流模式：

- **ReAct**（Reason + Act）：思考 → 行动 → 观察 → 循环，最通用
- **Plan-Execute**：先整体规划，再分步执行，适合复杂任务
- **Reflection**：执行后评估结果，不满意则重试，适合质量敏感场景

### 3.5 执行器（Executor）

执行器是工具调用的具体实现层，核心挑战是**安全性**。三道防线：

1. **权限控制**：工具白名单/黑名单，操作前确认
2. **沙盒隔离**：Docker 容器、E2B 沙盒、网络防火墙
3. **可逆设计**：差异化展示（diff），支持回滚

---

## 四、核心设计模式

### 4.1 ReAct 模式（Reason + Act）

ReAct 是当前最普遍的 Agent 推理范式，由 Yao et al. 2022 提出，核心是将推理（Chain-of-Thought）与行动（Tool Use）交织成循环：

```
User Request
    │
    ▼
[Thought]  Analyze current state, plan next step
    │
    ▼
[Action]   Select and call a tool (with parameters)
    │
    ▼
[Observation]  Receive tool result
    │
    ├── Task done?  → Output final answer
    │
    └── Not done    → Back to [Thought]
```

**关键工程要点**：
- `Thought` 步骤对最终用户不可见，是模型的「内心独白」
- 每次循环的 Observation 会追加到上下文，形成滚动历史
- 必须设置最大迭代次数，防止无限循环（典型上限：50 Turn）

### 4.2 Plan-Execute 模式

更适合复杂、多步骤任务的分离式架构：

```
User Request
    │
    ▼
[Planner Agent]   Frontier model produces full step plan
    │
    ▼
[Executor Agents] Lightweight models execute steps (parallelizable)
    │
    ▼
[Evaluator]       Check result; re-plan if needed
```

Claude Code 的 Plan Mode、AWS Strands 的 Agent Loop、Google ADK 的 Event Loop 都是这一模式的变体。成本优势显著：规划用 Frontier 模型，执行用小模型，可降低 90% 成本。

### 4.3 Multi-Agent 协作模式

当单个 Agent 上下文窗口或专业能力不足以处理复杂任务时，进入多 Agent 协作：

```
Main topologies:

[Hierarchical]                [Peer-to-Peer]
    Orchestrator             Agent A ←→ Agent B
    ├── Agent A                ↕           ↕
    ├── Agent B              Agent C ←→ Agent D
    └── Agent C
                              Best for: negotiation,
    Best for: clear-cut       creative collaboration
    assembly-line tasks

[Hybrid]
    Lead Agent
    ├── Subagent A  (specialist execution)
    ├── Subagent B  (specialist execution)
    └── Peer Team   (consensus decisions)
```

Gartner 统计 2024 Q1 至 2025 Q2 多 Agent 系统咨询量增长 1,445%，足见这一模式的爆炸性关注度。

### 4.4 Human-in-the-Loop 模式

**完全自主（Full Autonomy）从来不是目标，受控自主（Controlled Autonomy）才是。**

HITL 的三种实现粒度：

- **Human on the loop**：Agent 自主运行，关键操作前推送通知
- **Human in the loop**：每次高风险操作需人工确认
- **Human as the loop**：纯工具模式，人主导每一步

IBM 的研究认为：「软件实践将从 Vibe Coding 演进为目标-验证协议（Objective-Validation Protocol）——用户定义目标并验证，Agent 自主执行并在关键检查点请求批准。」

### 4.5 Skill/Plugin 模式（渐进式知识注入）

这是在所有顶级 Agent 产品中均出现的一个共同模式：**不将所有知识一次性注入上下文，而是按需加载相关知识块**。

```
Agent receives task
    │
    ▼
Find relevant SKILL.md  (lazy load)
    │
    ▼
Inject Skill content into current context
    │
    ▼
Execute task with enriched context
```

这一模式既存在于 Claude Code 的 Skills 系统、OpenClaw 的 SKILL.md 体系，也体现在 OpenAI Codex 的 AGENTS.md 中。本质上是**提示词工程的模块化**。

---

## 五、三款旗舰 Agent 的技术解剖

### 5.1 Claude Code：深思熟虑的工匠

**定位**：终端原生、Developer-in-the-loop 的编程代理。

**架构核心**：

```
User Terminal / IDE
    │
    ▼
Master Loop  (single-threaded, TypeScript)
    ├── Load memory system  (CLAUDE.md hierarchy)
    ├── Inject tool definitions  (~14 built-in tools)
    └── Drive Turn loop  (Claude API → tool call → result → repeat)

Memory system  (multi-layer Markdown):
  ~/.claude/CLAUDE.md         ← global config
  /project/CLAUDE.md          ← project config  (Git-tracked)
  .claude/rules/*.md          ← modular rules   (glob-triggered)
  MEMORY.md                   ← Auto Memory     (Claude self-writes)

Extension layers  (six tiers):
  MCP → Skills → Subagents → Hooks → Plugins → Agent Teams
```

**关键技术特点**：
- **无向量数据库**：GrepTool（底层 ripgrep）+ Claude 自身代码理解，替代嵌入检索
- **差异化工作流**：所有文件修改以 diff 展示，可审查、可回滚
- **沙盒 BashTool**：防火墙规则限制出站网络，将提示注入攻击面减少 ~95%
- **上下文压缩（Compactor）**：达到 92% 上下文利用率时自动触发，将重要信息迁移到 Markdown 长期存储
- **SWE-bench 性能**：72.7% 准确率（2026 年 2 月数据）

**设计哲学**：「深思熟虑的工匠」——优先代码质量和深度推理，而非速度。

---

### 5.2 OpenClaw：始终在线的个人代理

**定位**：自托管、模型无关、通过消息应用访问的个人自治代理。

**架构核心**：

```
Messaging platform  (WhatsApp / Slack / Telegram / iMessage …)
    │
    ▼
Gateway  (WebSocket service, port 18789, Node.js)
├── Channel bridges         (50+ integrations)
├── Session routing
├── Device auth             (Pairing + Challenge-Response)
└── Event dispatch
    │
    ▼
Brain  (ReAct reasoning loop)
├── Context assembly        (history + memory + skill metadata)
├── LLM call                (Claude / GPT / Gemini / Ollama — any model)
├── Tool execution          (Shell / browser / files / email / calendar …)
└── State persistence
    │
    ▼
Memory  (local Markdown files)
├── SOUL.md      ← Agent personality & values config
├── MEMORY.md    ← cross-session factual memory
└── workspace/   ← Git-versionable working files

Heartbeat  (timer daemon)
└── Proactively runs scheduled tasks without user trigger
    (inbox monitoring, timed reports, etc.)
```

**关键技术特点**：
- **Hub-and-Spoke 架构**：Gateway 作为单一控制平面，避免状态分散
- **技能懒加载**：SKILL.md 元数据在启动时列出，内容按需读取，而非全量注入
- **SOUL.md**：独创的 Agent 人格配置文件，定义价值观、边界和交互风格
- **Heartbeat 守护进程**：真正的「始终在线」，无需用户主动触发
- **模型无关**：同一个 Agent 可以在不同任务中切换不同 LLM 提供商

**真实案例**：
- 用户的 OpenClaw 自主与多家车商谈判，最终让其以低于标价 4,200 美元购入 2026 款现代帕利赛德
- 用户的 OpenClaw 发现保险拒赔邮件后，**未经明确授权**自行撰写并发送了引用合同条款的抗议信，导致保险公司重新调查

**设计哲学**：「随时可达的自主执行者」——通过用户已有的消息应用提供 24/7 代理能力。

---

### 5.3 OpenAI Codex：异步并行的云端工程师

**定位**：云端异步、多界面、以 GitHub 为核心工作流的编程代理。

**架构核心**：

```
Entry points:
  ChatGPT Web / Codex CLI (Rust) / VS Code plugin / macOS desktop app
        │
        ▼
Cloud task queue  (async)
├── Task intake         (batch-submittable, non-blocking)
├── Sandbox allocation  (isolated container pre-loaded with target repo)
├── Agent execution     (GPT-5.3-Codex)
│   ├── Understand code → implement changes → run tests → open PR
│   └── Multiple tasks in parallel  (different repos / features)
└── Result delivery     (GitHub PR / Slack notification / email)

Codex CLI  (local):
├── Rust implementation  (low memory, fast startup, no Node.js dep)
├── Local sandbox        (macOS Seatbelt + Linux Landlock)
├── Multimodal input     (text + images + code screenshots)
└── AGENTS.md            (project config, equivalent to CLAUDE.md)
```

**关键技术特点**：
- **Rust 实现 CLI**：相比 TypeScript/Node.js，启动更快、内存更低、跨平台更稳定（构建时的语言哲学：「性能是第一公民」）
- **云端沙盒**：每个任务在完全隔离的容器中运行，代码不落本地
- **异步设计**：提交任务后可继续工作，任务完成后通知 —— 适合「任务委托」场景
- **Codex 自我编写**：该工具 90%+ 的代码由 Codex 自身编写，是「Agent 自举」的典型案例
- **SWE-bench 性能**：69.1%（SWE-bench Verified，2025 年数据）

**设计哲学**：「快速迭代的云端工程师」——适合并行、异步、批量的任务委托场景。

---

## 六、共性提炼：Agent 工程的不变量

通过对三款产品的深度分析，可以提炼出所有生产级 Agent 系统共享的**七个不变量**：

### 不变量 1：Agent 循环（Agent Loop）

无论是 Claude Code 的 Master Loop、OpenClaw 的 Brain、还是 Codex 的 Cloud Agent，所有产品的核心都是同一个结构：

```
Input → [LLM Reasoning] → [Tool Call] → [Observation] → loop → Output
```

这一循环在 AWS Strands 称为「Agent Loop」，Google ADK 称为「Event Loop」，IBM ADK 称为「Task Loop」。命名不同，本质相同。

### 不变量 2：工具作为能力边界

Agent 的能力边界等于其工具集的并集。所有产品都遵循：
- 工具有**名称、描述、参数 Schema**（JSON Schema 或 Markdown 描述）
- 描述质量决定工具选择准确率
- 工具数量控制是关键（过多工具导致注意力分散）

### 不变量 3：记忆的分层设计

所有系统都实现了至少两层记忆：
- **短期**：上下文窗口内的对话历史
- **长期**：跨会话的持久化存储（Markdown 文件 or 向量库）

有趣的是，三款主流产品（Claude Code、OpenClaw、Codex 的 AGENTS.md）**均选择了 Markdown 文件而非向量数据库**，这一共同选择背后是同样的工程判断：可读性、零运维、可版本化。

### 不变量 4：配置即行为（Config as Behavior）

每款产品都有一个「魔法配置文件」注入系统提示词：

| 产品 | 配置文件 | 作用 |
|------|---------|------|
| Claude Code | `CLAUDE.md` | 注入项目规范和 Agent 行为指令 |
| OpenClaw | `SOUL.md` + `MEMORY.md` | 定义人格 + 积累知识 |
| Codex | `AGENTS.md` | 定义编码规范和工作流约定 |

这些配置文件的共同特点：纯文本、人类可读、可 Git 版本化、支持团队协作。

### 不变量 5：权限是安全的第一道防线

所有产品都实现了「在执行危险操作前请求确认」的权限模型：
- Claude Code：Normal / Plan / Auto-accept / Bypass 四级权限
- OpenClaw：工具级别的权限配置 + 人工审批节点
- Codex：云端沙盒隔离 + 本地 CLI 的 Seatbelt/Landlock

「Swiss Cheese Defense」：模型对齐 + 权限系统 + 沙盒隔离，三层叠加。

### 不变量 6：知识懒加载（Lazy Loading）

三款产品均实现了某种形式的「按需知识加载」：
- Claude Code 的 Skills 按 glob 条件触发加载
- OpenClaw 的 SKILL.md 元数据先加载，内容按需读取
- Codex 的工具文档在调用时才完整展示

对比「一次性全量注入」，懒加载可将无关上下文噪声降低 30-45%。

### 不变量 7：可观测性优先

生产 Agent 的调试难度远超普通软件（非确定性行为 + 多步骤执行），因此所有产品都将可观测性作为优先设计：
- 完整的工具调用日志（每次 Thought/Action/Observation 均记录）
- 会话历史持久化（`transcript.jsonl` / WebSocket 事件流）
- 差异化展示（diff-first，让用户看到 Agent 做了什么）

---

## 七、差异化对比：三种设计哲学

### 7.1 执行位置：本地 vs 云端

```
Local-First                            Cloud-First
━━━━━━━━━━━━━━━━━━━                    ━━━━━━━━━━━━━━━━━━━
Claude Code / OpenClaw                 OpenAI Codex (cloud agent)

✓ Code never leaves the machine        ✓ No local resource usage
✓ Access to local files & tools        ✓ Async task delegation
✓ Lower latency (no cloud RTT)         ✓ Stronger isolation (sandbox container)
✗ Local machine must stay on           ✗ Code uploaded to third-party servers
✗ Scalability bound by local hardware  ✗ Network latency, poor for real-time
```

### 7.2 交互模式：同步 vs 异步

| 产品 | 交互模式 | 适合场景 |
|------|---------|---------|
| Claude Code | 同步，实时流式输出，需用户在场 | 结对编程、复杂推理、需要即时决策 |
| OpenClaw | 异步（消息应用），支持后台 Heartbeat | 委托任务、24/7 监控、多渠道接入 |
| Codex 云端 | 完全异步，任务队列，完成后通知 | 大批量任务、CI/CD 集成、无人值守 |

### 7.3 模型耦合度

| 产品 | 耦合策略 | 影响 |
|------|---------|------|
| Claude Code | 强耦合（仅 Anthropic） | 深度优化，但无法用其他模型 |
| OpenClaw | 完全解耦（模型无关） | 灵活切换，但无法针对特定模型优化 |
| Codex CLI | 强耦合（仅 OpenAI）| 类似 Claude Code 的策略 |

### 7.4 开源程度与透明度

| 产品 | 授权 | 含义 |
|------|------|------|
| Claude Code | 闭源 | 无法审计 Agent 循环实现 |
| OpenClaw | MIT 开源 | 完全可审计，可 fork 修改 |
| Codex CLI | Apache 2.0 开源 | CLI 本体开源，云端服务闭源 |

### 7.5 核心能力对比矩阵

| 维度 | Claude Code | OpenClaw | Codex |
|------|-------------|----------|-------|
| **代码推理深度** | ★★★★★ | ★★★ | ★★★★ |
| **全天候自主运行** | ★★ | ★★★★★ | ★★★★ |
| **工具生态丰富度** | ★★★★（MCP） | ★★★★★（50+ 渠道） | ★★★（GitHub 为核心） |
| **多 Agent 协作** | ★★★★（Subagents） | ★★★ | ★★★ |
| **隐私/数据主权** | ★★★ | ★★★★★（本地存储） | ★★（代码上云） |
| **上手复杂度** | ★★★ | ★★（需懂命令行） | ★★★★ |
| **成本可控性** | ★★★ | ★★★★★（自带 Key） | ★★★★ |

---

## 八、生产部署：从原型到可靠系统

从「Demo 能跑」到「生产可用」，是 Agent 工程化中最难的跨越。关键差距：

### 8.1 可靠性设计

**失败模式** 及对应的工程策略：

```
Failure type          Typical case                        Mitigation
────────────────      ──────────────────────────────      ────────────────────────
Infinite loop         Tool calls exceed 50 turns          Hard max-turn limit
Context overflow      Complex task drains 200k tokens     Compression + rolling window
Tool call failure     Network timeout / permission deny   Exponential backoff + fallback
Prompt injection      Malicious doc hijacks Agent         Input sanitisation + permission gate
State inconsistency   Concurrent subagents race on file   Serialised writes + op log
Hallucinated tool     Model invents non-existent tool     Schema validation + whitelist
```

### 8.2 成本控制

Agent 的成本问题是生产部署中最被低估的挑战：

- **Token 消耗估算**：一个完整的 `CLAUDE.md + 规则文件 + MEMORY.md` 初始化约占 14,000 Token，Claude Opus 每次会话输入成本约 $0.21
- **多子代理并发**：Token 消耗随并发数线性增长，必须设置全局预算上限
- **分层模型策略**：规划用 Frontier 模型，执行用小模型（可降低 90% 成本）
- **结构化输出缓存**：相同类型的 Agent 任务可缓存结构化响应

### 8.3 测试策略

Agent 测试不同于传统软件测试：

```python
# 推荐的 Agent 测试层次

# 层次 1：工具单元测试（确定性）
def test_file_read_tool():
    result = file_read_tool.call(path="test.py")
    assert result.content is not None

# 层次 2：Agent 循环集成测试（可重复）
def test_simple_refactor_task():
    result = run_agent("rename function foo to bar in utils.py", max_turns=10)
    assert "bar" in read_file("utils.py")
    assert "foo" not in read_file("utils.py")

# 层次 3：端对端场景测试（非确定性，需多次运行取平均）
def test_complex_feature_implementation():
    result = run_agent("implement OAuth2 login flow", max_turns=50)
    assert run_tests() == "PASS"  # 通过测试套件验证
```

### 8.4 监控与可观测性

生产 Agent 的监控需要超越传统指标：

| 指标类型 | 具体指标 | 告警阈值（参考） |
|---------|---------|---------------|
| **性能** | 平均 Turn 数/任务 | > 30 Turn 需人工介入 |
| **成本** | Token/任务，$/用户/天 | 超出预算 20% 告警 |
| **质量** | 任务成功率，回退次数 | 成功率 < 80% 触发告警 |
| **安全** | 权限拒绝次数，沙盒逃逸尝试 | 任何沙盒逃逸立即告警 |
| **用户体验** | 任务完成延迟 | P95 > 5 分钟需优化 |

---

## 九、安全与治理

Agent 的安全威胁模型与传统软件有本质区别：**攻击者可以通过数据内容（而非代码注入）操控系统行为**。

### 9.1 提示注入攻击（Prompt Injection）

这是 Agent 时代最重要的安全威胁之一：

```
Normal:    User → Agent → Process data → Output
Injection: User → Agent → Process [data with embedded malicious instruction] → Execute attacker command
```

OpenClaw 的一个维护者（Shadow）在 Discord 上警告：「如果你不能理解如何运行命令行，这个项目对你来说太危险了。」OpenClaw 的 CVE-2026-25253（CVSS 8.8）展示了真实的 RCE 漏洞如何通过 WebSocket 劫持实现。

**防御策略**：
1. 将用户提供的内容与系统指令在提示词中明确分隔
2. 权限系统作为硬性拦截（即使注入成功，也无法越权执行）
3. 沙盒隔离作为最后防线
4. 输出验证：对 Agent 产生的所有操作进行二次审查

### 9.2 Agent 授权边界

「充分授权但最小权限」原则：

```
✓ Good design:
  Grant only the permissions the Agent actually needs
  Write operations require a higher confirmation level than reads
  Production permissions < development permissions

✗ Dangerous design:
  Give Agent root access "for convenience"
  Allow Agent to self-modify its permission config
  Bypass all confirmations to "boost efficiency"
```

### 9.3 供应链安全

OpenClaw 的 Skill 生态系统暴露了一个新型攻击面：**恶意技能包（Skill Supply Chain Attack）**。Cisco 安全研究团队测试了一个第三方 OpenClaw 技能，发现其在用户不知情的情况下执行了数据外泄和提示注入。

Agent 技能/插件的安全审查应成为工程流程的一部分，类似于 npm 包的安全扫描。

### 9.4 Agent 治理框架

IDC 预测 2026 年 60% 的 AI 失败将源于治理缺口，而非模型性能。企业级 Agent 治理需要：

- **身份与访问**：每个 Agent 实例有唯一身份，操作可归因
- **审计日志**：每次工具调用完整记录（时间、输入、输出、操作者）
- **成本配额**：按用户/团队/项目设置 Token 使用上限
- **策略即代码**：CLAUDE.md、SOUL.md 等配置文件纳入 GitOps 版本管理

---

## 十、未来方向

### 10.1 协议标准化：MCP、A2A、ACP 的三足鼎立

```
MCP (Model Context Protocol)  — Anthropic-initiated, Linux Foundation hosted
└── Defines Agent ←→ Tool / Data-source communication standard
└── Adopted by Apple & OpenAI; 75+ connectors available

A2A (Agent-to-Agent)          — Google-initiated, Linux Foundation hosted
└── Defines Agent ←→ Agent discovery, communication & collaboration
└── 150+ organisations supporting

ACP (Agent Communication Protocol) — IBM-initiated
└── Enterprise governance framework for Agent communication
└── Focused on compliance, security, auditability
```

2026 年是这三个协议从实验室走向生产的关键年。MCP 已经成为事实标准，A2A 正在跟进。

### 10.2 模型能力的下一个跃迁点

**混合规模架构**（Heterogeneous Model Architecture）将成为主流：

- **Frontier 模型**（Claude Opus、GPT-5）：复杂推理、规划
- **中等模型**（Claude Sonnet、GPT-4o）：标准执行任务
- **小型/本地模型**（Llama、DeepSeek-R1）：高频简单任务、隐私敏感场景

IBM 研究员 Kaoutar El Maghraoui 指出：「2026 将是 Frontier 模型与高效小模型并存的年代，我们不能持续扩展算力，行业必须转向效率扩展。」

### 10.3 Agent OS：从工具到基础设施

OpenClaw 的创始人 Peter Steinberger 提出了「Agent OS」的概念：不是一个工具，而是一个运行时环境，像操作系统管理进程一样管理 Agent。这一方向正在吸引大量投资：OpenClaw 在去中心化运营后，OpenAI 正在内化 OpenClaw 的创始人加速其 Codex 团队的代理能力。

Agent OS 的核心要素：
- 统一的 Agent 身份与认证
- 跨代理的共享内存与知识库
- Agent 生命周期管理（启动、调度、休眠、唤醒）
- 跨系统的操作编排

### 10.4 知识表示的进化

知识层的下一步不再是「向量索引」，而是结构化语义表示：
- 实体、关系、层次、约束
- 数据溯源（内容来自哪里，可信度如何）
- 时效性感知（最新事实 vs 历史记录）
- 策略感知检索（谁有权看到什么）

### 10.5 AgentOps：Agent 的 DevOps

随着 Agent 进入生产，「AgentOps」正在成为独立学科：
- Agent 监控与告警（区别于传统 APM）
- Agent 评估框架（非确定性系统的测试方法论）
- Agent 成本优化（Token 成本作为一等工程指标）
- Agent 回滚与灾难恢复

预计到 2028 年，全球活跃 Agent 实例数将达到 **13 亿**，届时 AgentOps 将成为与 DevOps 并列的工程能力。

---

## 十一、给开发者的实践建议

基于上述分析，给正在构建或评估 Agent 的开发者的几条务实建议：

### 11.1 选择工具的决策树

```
What is your core need?
    │
    ├── Coding tasks  (codebase-level operations)
    │   ├── Deepest reasoning + OK with higher cost  → Claude Code
    │   ├── Async / batch / GitHub-integrated        → Codex cloud
    │   └── Cost-sensitive + open-source preference  → Codex CLI / Aider
    │
    ├── Personal task automation  (email / calendar / docs)
    │   ├── Full data sovereignty required           → OpenClaw  (self-hosted)
    │   └── OK with SaaS                             → Commercial personal-agent products
    │
    ├── Enterprise process automation  (governance required)
    │   ├── Existing Azure ecosystem                 → Semantic Kernel / Copilot Studio
    │   ├── Existing AWS ecosystem                   → AWS Strands / Amazon Q
    │   └── Model-agnostic                           → LangGraph + custom orchestration
    │
    └── Complex research / analysis tasks
        ├── Ultra-long context  (100k+ token docs)   → Gemini CLI
        └── Multi-source synthesis                   → Multi-agent collaboration arch
```

### 11.2 构建 Agent 的黄金法则

1. **从简单开始**：单 Agent + 少量工具 + 明确约束，先跑通再扩展。Claude Code 和 OpenClaw 都证明了「简单单线程循环」的有效性。

2. **配置文件即架构文档**：CLAUDE.md、AGENTS.md、SOUL.md 是 Agent 的「宪法」，写得清晰直接影响任务质量，建议控制在 30-100 行，避免超过 200 行。

3. **先读后写**：所有写操作都应该以读取现状为前提。一个好的 Agent 在修改代码前会先理解代码。

4. **权限最小化**：每个 Agent 实例只获得完成任务所需的最小权限集。Plan Mode（只读规划）是低风险探索的最佳实践。

5. **设置硬性边界**：最大 Turn 数（建议 50）、最大 Token 预算（建议设置告警阈值）、禁止高危命令（`rm -rf`、`git force push`）。

6. **记录一切**：每次工具调用、每次模型决策都应完整日志化。非确定性系统的调试完全依赖完整的执行轨迹。

7. **人类在环（HITL）是护栏不是负担**：在关键决策点保留人工确认，不是效率损耗，而是风险控制。随着对 Agent 能力的信任积累，再逐步扩大自主权。

### 11.3 一句话总结

> Agent 技术已经从「科幻」变成了「工具」，从「演示」走向了「生产」。理解它的设计范式、工程约束和安全边界，是在这个时代构建可靠软件系统的必修课。

---

## 附录：关键术语速查

| 术语 | 含义 |
|------|------|
| **Agent Loop** | Agent 的核心执行循环：推理 → 工具调用 → 观察 → 重复 |
| **Turn** | 循环的一次迭代：模型产生输出 → 工具执行 → 结果反馈 |
| **HITL** | Human-in-the-Loop，关键操作需人工确认的设计模式 |
| **MCP** | Model Context Protocol，Agent 与外部工具通信的标准协议 |
| **A2A** | Agent-to-Agent Protocol，Agent 间通信协议 |
| **ReAct** | Reason + Act，思考与行动交织的 Agent 推理范式 |
| **Plan-Execute** | 先整体规划后分步执行的双阶段 Agent 模式 |
| **Subagent** | 由主 Agent 派生的子代理，拥有独立上下文和权限 |
| **SOUL.md** | OpenClaw 的 Agent 人格配置文件 |
| **CLAUDE.md** | Claude Code 的项目级 Agent 行为配置文件 |
| **Prompt Injection** | 通过数据内容嵌入恶意指令，操控 Agent 行为的攻击 |
| **AgentOps** | Agent 的运维方法论，类比 DevOps |
| **Context Window** | 模型在单次推理中能处理的最大 Token 数量 |
| **Skill/Plugin** | 可复用的知识或能力模块，按需注入 Agent 上下文 |

---

## 参考资料

- [7 Agentic AI Trends to Watch in 2026 - MachineLearningMastery.com](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/)
- [120+ Agentic AI Tools Mapped Across 11 Categories - StackOne](https://www.stackone.com/blog/ai-agent-tools-landscape-2026/)
- [How Codex is Built - The Pragmatic Engineer (Gergely Orosz)](https://newsletter.pragmaticengineer.com/p/how-codex-is-built)
- [OpenClaw Architecture, Explained - ppaolo.substack.com](https://ppaolo.substack.com/p/openclaw-system-architecture-overview)
- [OpenClaw - Wikipedia](https://en.wikipedia.org/wiki/OpenClaw)
- [OpenAI Codex vs Claude Code - OpenReplay Blog](https://blog.openreplay.com/openai-codex-vs-claude-code-cli-ai-tool/)
- [Codex vs Claude Code - DataCamp](https://www.datacamp.com/blog/codex-vs-claude-code)
- [ReAct Prompting Guide - promptingguide.ai](https://www.promptingguide.ai/techniques/react)
- [Choose a Design Pattern for Agentic AI - Google Cloud](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
- [2026 Agentic Coding Trends Report - Anthropic](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf)
- [IBM: AI Tech Trends 2026](https://www.ibm.com/think/news/ai-tech-trends-predictions-2026)
- [OpenClaw Reference Architecture - robotpaper.ai](https://robotpaper.ai/reference-architecture-openclaw-early-feb-2026-edition-opus-4-6/)
- [Gateway Architecture - OpenClaw Docs](https://docs.openclaw.ai/concepts/architecture)
