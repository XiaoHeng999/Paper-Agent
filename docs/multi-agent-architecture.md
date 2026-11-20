# 多 Agent 架构：现状、拆分依据与交接契约

> 状态：设计提案。本文先描述当前实现，再提出调整方向；提案中的角色和数据契约尚未实现。

## 1. 核心判断

项目当前是一个由 LangGraph 编排的固定工作流：`Search → Read → Analyse → WritingOutline → Writing → Reply`。各 Agent 在节点内执行专门的模型任务，节点负责检索、下载、持久化、状态更新和进度上报。Agent 之间没有直接对话，也不共享同一个 LLM 上下文窗口。

阶段拆分有工程价值，但不能仅因任务名称不同就拆成独立 Agent。值得拆分的边界应来自以下至少一项：

1. **可独立并行**：例如逐篇论文阅读、按子主题分析。
2. **上下文天然隔离**：例如单篇论文阅读只需要该论文，跨论文分析则需要多篇论文的可比较信息。
3. **目标或校验责任不同**：例如提出跨论文判断与独立核查该判断。
4. **工具或权限确实不同**：需由执行环境落实，而不只是写在 `AgentSpec` 中。

## 2. 当前实现怎样交接

`src/graph/graph.py` 固定了节点顺序；`src/graph/state_models.py` 定义共享 `State`。每个节点从 State 取字段、构造本次模型请求，再将结果写回 State。`AgentContext` 装配模型、工具等运行能力，不是跨 Agent 的对话记忆。

| 阶段 | 写入共享状态的主要结果 | 下阶段实际使用的内容 |
| --- | --- | --- |
| Search | `search_results`、`search_summary`、`search_output` | Read 读取筛选后的论文和原始请求 |
| Read | `read_results`、`read_summary` | Analyse 读取未被丢弃的单篇阅读结果，结合子主题信息分组 |
| Analyse | `analysis_report` | Outline 读取精简后的子主题与全局分析 |
| Outline | `writing_outline` | Writing 按章节、小节及证据映射安排写作 |
| Writing | `writing_sections`、`writing_report` | Reply 拼接最终 Markdown 与参考文献 |

State 保留的信息可能多于某次模型调用看到的信息。当前 handoff 风险主要发生在**构造下一次模型提示词**时：全文被提取成固定字段；Analyse 将单篇摘要和提取内容截短；Outline 再压缩分析报告。写作阶段可以按 ID 回取部分资料，但这无法保证上游未提取出的关键条件会被主动发现。`src/graph/analyse_node.py`、`src/agents/writingOutlineAgent.py` 和 `src/agents/writingAgent.py` 是这些边界的主要实现位置。

## 3. 推荐的最小角色划分

| 角色 | 边界与职责 | 拆分理由 |
| --- | --- | --- |
| Search Planner | 将研究问题拆成检索策略；基于证据缺口决定补搜。实际查询由检索服务执行 | 查询规划与证据处理使用不同输入和工具 |
| Paper Reader | 同一种工作单元按论文并行运行；抽取论文主张、前提、方法、实验条件和证据位置 | 可并行、上下文可按论文隔离 |
| Cross-paper Analyst | 对齐多篇论文的研究问题与实验条件，形成支持、差异、冲突和同质关系 | 必须在跨篇上下文中工作 |
| Evidence Verifier | 回查原文，核查重要主张与跨论文关系；可以退回证据不足的判断 | 与生成判断的角色目标不同，形成独立校验 |

编排器、去重、下载、存储、字段校验和状态转换仍是普通程序。Writing Outline 可以是按需的规划步骤；Writing 只有在用户请求综述或其他长文时才启动。这里的“并行 Paper Reader”表示同一角色处理多篇论文，并不要求每篇论文拥有不同的 Agent 实现或长期记忆。

## 4. 交接契约：共享研究对象，不传聊天摘要

每个阶段应写入版本化的结构化对象，而不是只交给下一阶段一段自然语言总结。最小契约包括：

```text
PaperRecord: paper_id、版本/DOI、来源、全文状态
EvidenceSpan: evidence_id、paper_id、原文位置、原文片段、解析版本
Claim: claim_id、paper_id、主张、前提/适用范围、evidence_ids、状态
AnalysisRelation: relation_id、claim_ids、关系类型、比较维度、理由、evidence_ids、核查状态
```

模型提示词可只携带相关对象的简要视图和稳定 ID；需要判断冲突或核查时，必须能通过 ID 回取原文。压缩只改变**本次阅读视图**，不能抹掉证据、前提、来源和处理版本。缺失字段应标为 `unknown`，不可当作“论文没有报告”。

推荐的执行路径：

```text
研究问题 → 检索规划 → 并行单篇阅读 → 论文判断卡与证据库
                                      ↓
                           跨论文关系分析 → 独立证据核查
                                      ↓
                           按任务生成比较、批判分析或综述
```

每次核查失败应能返回具体 `claim_id` 或 `relation_id` 及失败原因，触发针对性重读或补搜，而不是让整个链条重新概括一遍。

## 5. 如何判断拆分是否真的有效

在同一批论文和相同模型预算下，对照“现有串行多 Agent”“合并分析与大纲的简化流程”“带原文回查的结构化交接流程”。评估：

- 关键结论能否定位到真实原文；原文是否支持该结论。
- 实验条件、负面结果和限定语的保留率。
- 同质论文、条件差异和真实冲突的判定准确率。
- 运行时间、模型调用次数、token 成本与失败恢复成本。

只有新增角色带来可测的质量、并行或隔离收益，才继续保留该拆分。相关的跨论文对象与核查规则见 [对抗与多论文分析设计](./adversarial-multi-paper-analysis.md)。
