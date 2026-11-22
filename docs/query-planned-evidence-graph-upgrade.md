# Query-Planned Evidence Graph：Paper-Agent 阅读链路优化实施方案

本文给出一套可以直接应用于当前 Paper-Agent 的渐进式升级方案。目标不是一次性重写整个研究工作流，而是优先解决当前最影响质量的几个问题：

1. `topic` 同时承担用户意图、搜索、阅读、召回和写作目标；
2. 指定论文时跳过 Search Agent，也缺少独立的阅读问题规划；
3. PDF 按页和固定字符切分，证据单元缺少结构和关系；
4. 检索以独立 chunk 相似度为主，无法稳定组装条件完整的证据；
5. 图像、Caption、OCR 和附近正文没有作为同一对象的多种视图协同检索。

方案借鉴 GraphMemix 的核心思想：相似度只负责找到入口，结构关系负责补齐上下文，节点判断负责识别直接证据，边判断负责识别增量信息，最终选择的是完整证据包而不是若干孤立 Top-K 文本。

参考论文：<https://arxiv.org/pdf/2608.26983>

相关现状审计见 [PDF 阅读链路中的 Topic、Query、解析、切分与检索问题分析](./pdf-query-chunking-retrieval-analysis.md)。

---

## 当前实现状态（2026-11-22）

当前已经完成可独立验证的第一阶段、第二阶段的文本结构基础，以及第三阶段的首个可运行版本：

- 已增加确定性的 `PaperReadingPlanner`，把普通 prompt 分为 `open_ended`、`guided`、`targeted`，并生成有上限的阅读子问题；
- 已将 ReadingPlan 接入逐篇召回、Reader 提示、claim 绑定、缓存版本和报告覆盖状态；
- 已由“单个原始 topic 做一次 Top-K”改为“逐子问题召回后轮转合并”，embedding 不可用时保留词法回退；
- 已将默认文本切分器升级为 `StructureAwareChunker`，优先保留页码、章节、段落和句子边界，并记录 overlap、父段及前后邻接；
- 已让证据存储和上下文扩展读取显式前后关系，同时保留旧数据的位置回退路径；
- 已实现 dense、词法、章节、精确编号和 evidence requirement 五路召回，并用 RRF 融合；
- 已对高分 seed 执行有界前后邻接和同父节点扩展，保留每个候选的召回通道及相对 Anchor 关系；
- 已实现覆盖式贪心 Evidence Packet，在字符和节点预算内优先补齐条件、限制与负面结果；
- 已将 Packet 接入 Reader、缓存和报告，Reader 只能看到当前调用实际提供的 Packet 证据；
- 已对仍有缺口的 Packet 执行至多一轮 gap dense retrieval，并保留未满足需求和 repair query；
- 已补充规划、切分、关系、混合召回、组包和工作流接入测试。

尚未完成的部分不能视为已经具备：layout-aware PDF parser、Caption/表格/公式对象化、OCR/图像 embedding 多视图、模型化 Node/Increment Verifier，以及完整视觉对象检索。当前 requirement 识别仍是确定性候选分类，不等于事实核查；只有 Reader 产生关联 claim 且 Packet 没有需求缺口时，子问题才会标为 `covered`。

---

## 1. 最重要的六项改造

建议按以下优先级实施：

| 优先级 | 改造 | 直接解决的问题 |
| --- | --- | --- |
| P0 | 新增独立 `PaperReadingPlanner` | 指定 PDF 时没有问题拆解；模糊 prompt 无法形成有效 query |
| P0 | 将 `topic` 拆成用户意图、分析维度、子问题和检索 query | Topic/Query 语义过载 |
| P0 | 引入结构化证据单元和确定性关系边 | 固定字符切分、上下文断裂、Chunk 关系丢失 |
| P1 | 对每个子问题进行多视图混合召回和一跳关系扩展 | 单一向量 Top-K 漏掉低相似度互补证据 |
| P1 | 分离“节点直接支持”和“相对 Anchor 的增量作用” | 重复证据过多、条件和反证不足 |
| P1 | 按证据需求覆盖度选择 Evidence Packet | 检索到相关文本但缺少实验条件、基线或限制 |

第一版不建议立即实现 GraphMemix 的完整森林组合优化。应先实现可解释的覆盖式贪心选择，建立评测后再判断复杂优化是否必要。

---

## 2. 目标 Agent 边界

当前 Search Agent 负责从用户主题生成论文数据库检索计划。这个职责应保留，但不应让它兼任指定论文的阅读规划。

建议形成以下职责边界：

```text
SearchAgent
  只负责：没有指定论文时，生成检索子主题并寻找论文

PaperReadingPlanner
  负责：理解用户阅读意图、识别论文类型、生成分析子问题和证据需求

EvidenceRetriever
  负责：多视图召回、结构关系扩展、证据候选组织

EvidenceSelector
  负责：直接支持判断、增量关系判断、Evidence Packet 选择

PaperReader
  负责：只基于已提供的 Evidence Packet 提取原子 claims

EvidenceVerifier
  负责：核查 claim、寻找限定/反证、输出缺口和修复请求

Writer
  负责：依据已核查 claims 和依赖关系生成报告，不自行检索新事实
```

关键原则是：

> 找论文和读论文是两个不同的规划问题，不能因为用户已经指定 PDF 就跳过所有问题规划。

---

## 3. 第一项改造：新增 PaperReadingPlanner

### 3.1 何时运行

Planner 应在以下信息可用后运行：

- 用户原始 prompt；
- 论文标题和摘要；
- 论文目录或粗略章节列表；
- 输出模式和用户约束。

它不需要先读取整篇论文，也不应依赖已经生成的 claims。这样可以避免“先读到什么，就只围绕什么提问题”的首批证据偏差。

推荐流程：

```text
PDF 下载与轻量解析
  ↓
生成 title / abstract / outline / paper_type_hint
  ↓
PaperReadingPlanner
  ↓
得到 ReadingPlan
  ↓
按子问题检索全文证据
```

### 3.2 如何处理无 query、模糊 query 和明确 query

Planner 应把请求分为三类：

#### 开放式阅读

用户输入类似：

```text
阅读这篇论文
总结一下
分析这篇文章
```

处理方式：生成完整的默认分析维度，并明确记录 `mode=open_ended`。不能把“阅读这篇论文”直接当作 embedding query。

#### 模糊定向阅读

用户输入类似：

```text
看看这个方法是否有效
分析创新点
这篇论文靠谱吗
```

处理方式：保留用户目标，同时补齐回答该目标必需的机制、实验条件、替代解释和限制问题。

#### 明确定向问题

用户输入类似：

```text
该方法的推理加速是否在相同硬件、精度和 batch size 下成立？
```

处理方式：将用户问题原子化，给予最高优先级；默认全局问题只保留支持理解和防止误判所需的部分。

### 3.3 固定分析维度，而不是固定字面问题

建议固定以下六个分析维度：

1. `problem_and_scope`：问题、动机、贡献和适用范围；
2. `method_mechanism`：输入、操作、输出、关键模块和信息流；
3. `assumptions`：理论、数据、环境和实现前提；
4. `experimental_design`：数据、基线、指标、预算、硬件和统计设置；
5. `causal_validation`：消融、机制归因、替代解释和验证逻辑；
6. `limitations_reproducibility`：失败场景、外部有效性和复现信息。

固定的是覆盖维度，具体问题必须根据论文类型和用户目标动态生成。例如理论论文不应强制寻找不存在的训练硬件，数据集论文则必须增加标注协议、泄漏和分布偏差问题。

### 3.4 建议的数据结构

可在 `src/research/models.py` 或新的 `src/research/planning.py` 中增加：

```python
class EvidenceRequirement(Record):
    kind: Literal[
        "problem", "claim", "method_step", "equation", "assumption",
        "dataset", "baseline", "metric", "budget", "hardware",
        "result", "ablation", "negative_result", "limitation",
        "implementation", "figure", "table"
    ]
    required: bool = True
    description: str = ""


class ReadingSubquery(Record):
    id: str
    source: Literal["user", "user_derived", "default_lens", "gap_repair"]
    lens: Literal[
        "problem_and_scope", "method_mechanism", "assumptions",
        "experimental_design", "causal_validation",
        "limitations_reproducibility"
    ]
    question: str
    retrieval_queries: list[str]
    evidence_requirements: list[EvidenceRequirement]
    priority: int = Field(ge=1, le=5)
    status: Literal["pending", "covered", "partial", "unanswered"] = "pending"


class ReadingPlan(Record):
    paper_id: str
    mode: Literal["open_ended", "guided", "targeted"]
    paper_type: Literal[
        "algorithm", "system", "theory", "dataset_benchmark",
        "survey", "empirical_study", "unknown"
    ]
    user_prompt: str
    user_questions: list[str]
    subqueries: list[ReadingSubquery]
    warnings: list[str] = Field(default_factory=list)
```

### 3.5 Planner 输出约束

为避免 Planner 自己制造新的 query 膨胀，应加硬约束：

- 每篇默认生成 4–6 个子问题；
- 明确用户问题可以增加，但总数第一版不超过 8；
- 每个子问题只表达一个可核查目标；
- 用户问题必须出现在 `user_questions`，不能被改写后丢失；
- 每个子问题必须声明至少一个 evidence requirement；
- 重复子问题必须合并；
- 不得根据标题和摘要提前断言论文结论；
- `retrieval_queries` 是检索改写，不等于新的用户意图。

### 3.6 需要修改的代码位置

- `src/agents/contracts.py`：保留 `ResearchRequest.topic` 作为兼容字段，但把它解释为原始用户 prompt；
- `src/research/models.py`：增加 ReadingPlan 相关模型；
- 新建 `src/research/planning.py`：Planner 调用、校验和默认计划兜底；
- `src/research/workflow.py`：在生成 PaperCard 前为每篇论文创建或恢复 ReadingPlan；
- `src/research/incremental.py`：将 ReadingPlan 版本加入缓存依赖；
- 报告产物：保存原始 prompt、ReadingPlan、子问题状态和未满足需求。

---

## 4. 第二项改造：拆开 Topic 和运行时 Query

### 4.1 不再直接使用 request.topic 做所有事情

保留当前入口兼容性：

```python
ResearchRequest.topic
```

但进入研究节点后应标准化成：

```text
raw_user_prompt
reading_mode
user_questions
analysis_lenses
reading_subqueries
retrieval_queries
evidence_requirements
output_goal
```

不同阶段只能读取自己需要的字段：

| 阶段 | 应使用的内容 |
| --- | --- |
| Search Agent | 原始主题、年份、来源等搜索约束 |
| Reading Planner | 原始 prompt、标题、摘要、目录、输出目标 |
| Evidence Retriever | 当前子问题的 retrieval queries 和 evidence requirements |
| Reader | 当前子问题、Evidence Packet、用户原始问题 |
| Verifier | claim、证据包、缺口和反证 query |
| Writer | 用户问题、已核查 claims、ReadingPlan 覆盖状态 |

### 4.2 不允许内部 query 偷偷改变用户目标

所有内部 query 必须带 `source` 和 `parent_id`。例如：

```json
{
  "id": "rq-17",
  "source": "gap_repair",
  "parent_id": "subquery-experimental-fairness",
  "query": "hardware precision batch size latency baseline implementation",
  "reason": "速度结果缺少硬件和 batch size 条件"
}
```

这样报告可以区分：

- 用户实际问了什么；
- Planner 为回答问题补了什么；
- Verifier 因证据不足又查了什么。

---

## 5. 第三项改造：结构化证据单元，而不是固定字符 Chunk

### 5.1 三层数据模型

不要让同一个 chunk 同时承担原始存储、检索和模型上下文三种职责。建议拆成三层。

#### 原始文档对象层

保存稳定、可定位的对象：

```text
Heading
Paragraph
Equation
Figure
FigureCaption
Table
TableCaption
AlgorithmBlock
Footnote
Reference
```

每个对象至少保存：

- `paper_id`；
- `node_id`；
- `node_type`；
- `page_start/page_end`；
- `bbox`；
- `section_path`；
- `reading_order`；
- `raw_text`；
- `source_version`；
- 可选图片裁剪或页面区域引用。

#### 检索单元层

一个检索单元可以对应：

- 一个完整段落；
- 一个公式及前后解释；
- 一张图、Caption、OCR 和附近讨论；
- 一个表格、标题、表头和结果讨论；
- 一个过长段落中按句子边界拆出的子段。

#### Query-time Evidence Packet

这是最终交给 Reader 的动态上下文，例如：

```text
Anchor：Table 3 的延迟结果
├── Experimental Setup 的硬件说明
├── Baseline 配置
├── Table 3 Caption 和表头
├── 正文结果解释
└── Limitation 中的适用条件
```

### 5.2 第一版可落地的切分规则

不必第一步就引入完整学术解析平台。可以先替换最危险的固定字符硬切：

1. 按标题和段落块切分；
2. 过长段落按句子边界切分；
3. 子段之间保留一到两句 overlap；
4. 不在公式、列表、Caption、表格行中间切；
5. 不跨章节打包；
6. 被拆分的段落增加 `continuation_of`；
7. 公式默认绑定前一个定义段和后一个解释段；
8. Figure/Table Caption 独立保存，同时与视觉对象和讨论段落建立边；
9. 页码和 bbox 始终来自原始对象，不从拼接文本猜测。

建议使用 token 预算而不是固定字符预算。第一版可以把检索单元控制在约 450–700 tokens，但具体范围必须通过评测调整，不能作为永久常量。

### 5.3 解析后端建议

当前 `pypdf` 适合作为快速文本兜底，但不应是唯一结构来源。建议保持现有 `BasePdfParser` 接口并新增 layout-aware backend：

- 短期：PyMuPDF 提取 block、bbox、字体和图片区域；
- 中期：对学术论文评估 GROBID、Docling 或其他结构解析后端；
- 扫描页：OCR 结果作为一种视图，不覆盖原始页面；
- 所有后端最终归一化为统一的 `DocumentNode + DocumentEdge` 模型。

不得直接删除现有 `paper.md` 和 `chunk.json`。迁移期应新增：

```text
document.json
evidence-graph.json
```

旧文件继续用于兼容和回退。

### 5.4 需要修改的代码位置

- `src/utils/read_utils/pdf_parsers.py`：扩展解析结果，支持布局块和对象类型；
- `src/utils/read_utils/read_fulltext.py`：除 Markdown 外保存结构化文档；
- `src/utils/read_utils/chunkers.py`：增加 StructureAwareChunker，保留 PageChunker 作为回退；
- `src/research/storage.py`：从结构化检索单元生成 EvidenceSpan；
- `src/research/provenance.py`：把 document/graph 版本加入来源清单。

---

## 6. 第四项改造：建立轻量 Evidence Graph

### 6.1 第一版只做可靠关系

先建立确定性和可解释的关系：

```text
parent_section
previous
next
continuation_of
caption_of
discusses_figure
discusses_table
defines_symbol
explains_equation
same_experiment
appendix_supports
```

再增加 mutual-KNN 语义边。显式结构边的可靠性应高于纯语义边。

建议模型：

```python
class DocumentEdge(Record):
    source_id: str
    target_id: str
    relation: str
    confidence: float = Field(ge=0, le=1)
    origin: Literal["layout", "reference", "rule", "semantic", "model"]
    source_version: str
```

### 6.2 为什么不能只保留 previous/next

线性邻接只能找回局部连续文本，不能表达：

- 结果引用了哪张表；
- 消融对应哪个模块；
- 公式依赖哪些定义；
- 附录补充正文哪个实验；
- 限制条件约束哪条 claim。

Evidence Graph 的目标不是构建一个庞大知识图谱，而是支持检索阶段可靠地从直接命中扩展到必要上下文。

### 6.3 扩展策略

第一版采用有界一跳扩展：

```text
Top-N seeds
  ↓
按关系白名单扩展一跳
  ↓
每个 seed 每种关系最多取若干节点
  ↓
候选总数超过上限时按边可靠性和初始召回分裁剪
```

只有以下情况再允许第二跳：

- 第一跳命中了 Caption，需要继续到视觉对象；
- 第一跳命中了结果讨论，需要继续到实验设置；
- 第一跳命中了公式解释，需要继续到公式或定义；
- Evidence Requirement 明确仍有缺口。

禁止无界沿同章节或语义边扩展。

---

## 7. 第五项改造：多视图混合召回

### 7.1 一个对象保留多个检索视图

以 Figure 为例：

```text
Figure node
├── image embedding
├── caption embedding
├── OCR embedding
├── section-path text
├── nearby discussion text
└── figure number / page / metadata
```

以 Table 为例：

```text
Table node
├── caption
├── headers
├── normalized rows
├── OCR fallback
├── nearby analysis
└── page image crop
```

不要预先把这些视图合并成一段长文本。召回时可以通过任意视图命中同一个 canonical node，再在后续阶段加载完整对象。

### 7.2 第一版混合召回组成

对每个 ReadingSubquery 并行执行：

1. Dense text retrieval；
2. BM25 或关键词检索；
3. 标题和章节路径匹配；
4. Figure/Table/Equation 编号精确匹配；
5. Caption 和 OCR 召回；
6. 有视觉 embedding 时执行图像召回。

召回结果统一映射回 canonical node，做去重和分数归一化。

不要在第一版固定一个看似精确的权重公式。可以先采用 Reciprocal Rank Fusion，减少不同分值尺度的校准成本：

```text
fused_score(node) = Σ 1 / (c + rank_view(node))
```

然后通过离线评测决定是否需要学习权重或使用 reranker。

### 7.3 与现有 EvidenceIndex 的结合

当前 `src/research/retrieval.py` 可以继续承担 dense retrieval，但需要扩展为：

- 每个 canonical node 可以有多个 view record；
- 搜索结果回填 authoritative canonical node，不能直接信任索引副本；
- 索引 metadata 增加 `node_id`、`view_type`、`section_path` 和对象类型；
- 查询接口返回 fusion 前的各路命中，供诊断和评测；
- 原有按论文和来源版本隔离继续保留。

---

## 8. 第六项改造：直接支持和增量作用分离

这是 GraphMemix 最值得直接借鉴到 Agent 的部分。

### 8.1 Node Verifier

Node Verifier 只回答：

> 这个候选单独看，是否直接帮助回答当前子问题？

输出建议：

```json
{
  "node_id": "...",
  "direct_support": 0,
  "supported_requirements": ["result", "hardware"],
  "reason": "..."
}
```

使用 listwise 调用，一次评估一批候选，避免为每个节点单独调用模型。

### 8.2 Edge/Increment Verifier

它只回答：

> 相对于某个高置信度 Anchor，这个候选提供什么增量作用？

关系角色采用：

```text
new_fact
clarification
corroboration
redundant
conflict
irrelevant
```

对论文阅读还可以增加：

```text
condition
method_explanation
experimental_context
limitation
```

但第一版最好保持角色数有限，避免输出不稳定。`condition` 和 `limitation` 也可以先归入 `clarification`，通过 evidence requirement 表示具体类型。

### 8.3 为什么要分成两个调用

一个节点可能：

- 单独看不直接回答问题；
- 但它补充了 Anchor 缺失的硬件条件；
- 或者指出 Anchor 结果只在特定数据集成立；
- 或者只是重复 Anchor 的同一句话。

直接相关性和条件增量是不同判断，合并到一个总分会让低相似度但关键的条件证据被丢弃。

Node Verifier 和 Increment Verifier 可以并行执行，不需要串行增加关键路径。

---

## 9. Evidence Packet 选择：先用覆盖式贪心，不急于完整森林优化

### 9.1 选择目标

每个子问题都有 evidence requirements。选择器应同时考虑：

- 对当前子问题的直接支持；
- 新覆盖的 evidence requirement；
- 与 Anchor 的互补作用；
- 是否为限定、反证或失败信息；
- 与已选节点的重复程度；
- token 和图像预算；
- 关系可靠性。

可采用以下概念目标：

```text
score(candidate) =
    direct_support
  + new_requirement_coverage
  + trusted_incremental_value
  + contradiction_or_limitation_bonus
  - redundancy
  - relation_uncertainty
  - context_cost
```

### 9.2 贪心选择流程

```text
1. 选择最高直接支持节点作为 Anchor
2. 优先选择能够填补尚未满足 requirement 的节点
3. 对相同 requirement，选择增量最高且重复最低的节点
4. 强制保留可信 conflict/limitation，不让正向证据淹没反证
5. 达到 token、图像或节点预算后停止
6. 输出未满足 requirement 列表
```

### 9.3 Packet 序列化

交给 Reader 时按照：

```text
子问题
证据需求
Anchor
与 Anchor 相关的条件/解释/佐证/冲突
第二条独立证据链
未满足需求
```

每个节点必须附带：

- evidence ID；
- paper/page/section；
- node type；
- 与 Anchor 的关系；
- 来源版本；
- 若为视觉对象，附原图或裁剪版本。

### 9.4 何时再实现 Evidence Forest

满足以下条件后再考虑完整森林优化：

1. 结构化节点和关系准确率已经稳定；
2. 有证据级标注集；
3. 贪心选择相对 Top-K 已有明确收益；
4. 发现贪心方法经常因为局部最优漏掉成组证据；
5. 有足够候选规模使组合优化有实际价值。

否则完整森林优化只会精确地优化不可靠节点和边。

---

## 10. Gap-driven Reread：让缺口驱动下一轮 Query

Reader 和 Verifier 完成一轮后，不应只判断 claim 是否通过，还应更新子问题状态。

示例：

```json
{
  "subquery_id": "speed-fairness",
  "status": "partial",
  "covered": ["result", "baseline", "hardware"],
  "missing": ["batch_size", "precision"],
  "repair_queries": [
    "batch size precision latency experimental setup",
    "inference configuration appendix hardware"
  ]
}
```

只对缺口执行下一轮召回：

```text
缺失 requirement
  ↓
生成 gap query
  ↓
混合召回 + 关系扩展
  ↓
补充 Evidence Packet
  ↓
重新核查受影响的 claim
```

建议最多一到两轮，受现有 `max_repair_rounds` 控制。未找到证据时必须记录“未报告/未找到”，不能填充推测。

---

## 11. 视觉链路调整

当前视觉页主要通过页面启发式提前选择。建议逐步改为“低成本建目录，按子问题按需渲染”。

### 11.1 解析阶段

先建立视觉目录，不立即把所有候选页发送给模型：

- Figure/Table 编号；
- Caption；
- page 和 bbox；
- OCR；
- 附近正文；
- 页面图像对象面积；
- source/image version。

### 11.2 检索阶段

只有以下对象进入视觉候选：

- 被 retrieval query 命中；
- 被文本证据显式引用；
- 对未满足 evidence requirement 有帮助；
- Verifier 判断必须回看原图。

### 11.3 模型输入阶段

优先发送对象级裁剪，同时保留整页作为定位上下文：

```text
Figure/Table crop
+ Caption
+ page number
+ surrounding paragraph
+ full-page thumbnail when necessary
```

视觉转录仍然不是独立最终证据，核查时必须绑定相同 `image_version` 的原图。

需要调整：

- `src/research/visual.py`：拆分 catalog、on-demand render 和 verification；
- `verification_images()`：从页级依赖扩展到对象级依赖；
- EvidenceSpan/DocumentNode：增加 `bbox`、`visual_object_id`、`caption_id`。

---

## 12. 工作流中的具体接入顺序

建议将 `src/research/workflow.py` 的单篇处理改造成以下逻辑：

```text
prepare_input
  1. 校验来源版本
  2. 加载结构化文档和 Evidence Graph
  3. 建立多视图索引
  4. 保存轻量 outline

plan_reading
  5. 根据用户 prompt + title/abstract/outline 生成 ReadingPlan
  6. 校验并缓存 ReadingPlan

retrieve_per_subquery
  7. 对每个子问题执行混合召回
  8. 从 seeds 沿可靠关系有界扩展
  9. Node/Increment Verifier 并行判断
  10. 选择 Evidence Packet

read_and_verify
  11. Reader 从 Packet 抽取 claims
  12. claim 绑定 subquery_id 和 requirement IDs
  13. Verifier 核查并输出缺口
  14. 必要时执行一轮 gap-driven reread

synthesize
  15. 汇总已覆盖和未覆盖子问题
  16. 生成方法链、实验逻辑、假设、限制和用户问题答案
  17. 报告显式展示证据缺口和视觉未核查范围
```

### 缓存键必须包含

- 原始 PDF/source version；
- parser 和 chunker policy version；
- Evidence Graph version；
- embedding model/version；
- Planner policy version；
- ReadingPlan version；
- 用户原始 prompt 和约束；
- selector policy version；
-视觉对象版本。

这样修改 Planner、解析器或关系策略时，旧结果不会被错误复用。

---

## 13. 分阶段实施计划

### Phase 1：Reading Planner，保持现有 Chunk

目标：先解决 Query 过载，不动 PDF 解析主链。

工作项：

1. 增加 ReadingPlan 模型；
2. 新增 `PaperReadingPlanner`；
3. 将模糊 prompt 分类为 open-ended/guided/targeted；
4. 为每篇生成 4–8 个子问题；
5. 使用每个子问题分别调用现有向量检索；
6. claims 增加 `subquery_id`；
7. 报告增加子问题覆盖情况。

这一阶段可以直接验证“问题规划是否比直接使用 topic 更好”。

### Phase 2：StructureAwareChunker 和 Evidence Graph

目标：解决固定字符切分和弱邻接。

工作项：

1. 增加 layout-aware parser；
2. 保存 `document.json`；
3. 实现段落/句子边界切分和 overlap；
4. 识别标题、Caption、公式和表格对象；
5. 保存确定性关系边；
6. 保留 PageChunker 回退路径；
7. 对解析失败和扫描 PDF 做明确降级。

### Phase 3：混合召回和 Evidence Packet

目标：从 Top-K chunk 转向完整证据包。

工作项：

1. Dense + lexical + exact reference 融合；
2. 一跳结构扩展；
3. canonical node 去重；
4. evidence requirement 覆盖式选择；
5. 输出 Packet trace；
6. 对未满足 requirement 生成 gap query。

### Phase 4：Node/Increment Verifier 和多模态对象检索

目标：抑制重复，找回低相似度条件证据和视觉证据。

工作项：

1. listwise Node Verifier；
2. listwise Increment Verifier；
3. Figure/Table 多视图索引；
4. 对象级裁剪和按需渲染；
5. conflict/limitation 强制保留策略；
6. 并行调用与预算统计。

### Phase 5：评估是否需要完整 Forest Optimizer

只有离线错误分析证明贪心选择经常漏掉成组证据时，再增加最大权森林或局部交换优化。

---

## 14. 测试方案

### 14.1 Planner 测试

至少覆盖：

- 空泛 prompt 生成默认分析维度；
- 明确问题被原样保留并拆成原子子问题；
- 用户问题优先级高于默认问题；
- 理论、系统、算法、数据集和 Survey 使用不同问题模板；
- 不超过最大子问题数量；
- Planner 失败时使用确定性默认计划，而不是回到原始 topic 直接召回。

### 14.2 切分和结构测试

- 不在句子中间硬切；
- 公式和解释保持绑定；
- Caption 与 Figure/Table 建边；
- 跨页连续段落有 `continuation_of`；
- page/bbox/source version 正确；
- 双栏 fixture 的 reading order 可审计；
- 解析失败时回退 PageChunker 并给出 warning。

### 14.3 检索测试

构造低相似度互补证据案例：

```text
Anchor：结果表明延迟降低 30%
互补证据：所有实验运行在更低数值精度下
重复证据：摘要再次声称延迟降低
冲突证据：附录中长序列场景没有加速
```

要求：

- 互补条件通过关系扩展进入候选；
- 重复证据被降权；
- 冲突/限制被保留；
- Evidence Packet 覆盖结果、硬件/精度和限制；
- 未满足需求明确输出。

### 14.4 多模态测试

- query 只与图像内容相关时可以通过 image view 命中；
- query 包含表格数值时可以通过 OCR/header view 命中；
- Caption 命中后能扩展到原图和讨论段落；
- image version 不匹配时拒绝复用旧观察；
- 未提供原图时不能把视觉依赖 claim 标成 verified。

### 14.5 恢复和版本测试

- 用户 prompt 改变导致 ReadingPlan 和检索缓存失效；
- parser policy 改变导致 document/graph/chunk 重建；
- embedding 模型改变不混用旧索引；
- Planner policy 改变只重做下游计划和分析，不必重新下载 PDF；
- 单篇解析失败不影响其他论文。

---

## 15. 评测指标和验收标准

不要只看最终回答是否“看起来更好”。应分别评估：

### Planner

- 用户问题保留率；
- 子问题去重率；
- 默认分析维度覆盖率；
- 不同论文类型的问题适配准确率。

### Retrieval

- Evidence Recall@Budget；
- 低相似度互补证据恢复率；
- 条件完整率；
- 反证/限制召回率；
- 重复证据比例；
- 每个 requirement 的覆盖率。

### Parsing and linking

- 章节识别准确率；
- 段落边界准确率；
- Figure/Table-Caption 绑定准确率；
- 正文引用到视觉对象的解析准确率；
- page/bbox 定位正确率。

### Claims and reports

- claim 引用精度；
- claim 条件完整率；
- 跨 chunk 限定条件遗漏率；
- 用户问题回答完整率；
- 未知与未报告的正确区分率；
- 专家判断一致率。

### Cost

- 每篇调用次数；
- 输入/输出 token；
- embedding 次数；
- 视觉页/对象数量；
- 端到端延迟；
- 缓存复用率。

### 第一阶段建议验收门槛

先用一组人工标注论文比较当前流程和新流程。至少应满足：

- 用户显式问题全部进入 ReadingPlan；
- 默认六维中适用于论文类型的维度均有对应子问题；
- 在相同上下文预算下，实验条件完整率和限制召回率明显高于当前 Top-K；
- 重复证据比例下降；
- 未核查视觉依赖不会被误标为 verified；
- 模型调用和延迟增量可解释且受预算控制。

具体数值应在建立基准集后确定，不能先拍脑袋设定通过率。

---

## 16. 推荐的最小可行版本

如果只做一轮最小改造，建议实施以下组合：

1. 增加 `PaperReadingPlanner`；
2. 固定六个分析维度，按论文类型生成 4–8 个子问题；
3. 用户问题原子化并设置最高优先级；
4. 每个子问题独立使用现有 EvidenceIndex 召回；
5. 召回结果补充前后邻居、同页 Figure/Table Caption 和实验设置关键词片段；
6. 按 evidence requirement 做简单覆盖式选择；
7. claims 绑定 `subquery_id`；
8. 报告列出每个子问题的已覆盖、部分覆盖和未回答状态。

这个版本暂时不依赖新的 PDF 后端，也不需要完整 Evidence Forest，但已经能够验证最核心假设：

> 先规划论文阅读问题，再围绕每个问题组装证据，是否比直接把原始 topic 用作全文检索 query 更可靠。

如果该假设通过评测，再投入结构解析、多视图索引和增量关系验证。

---

## 17. 最终建议

当前最应该避免的是同时开展大规模 PDF parser 替换、完整知识图谱、复杂森林优化和全部 Agent 重写。这样难以确定质量变化来自哪里，也会破坏现有的来源追踪和恢复能力。

推荐顺序是：

```text
先拆 Topic/Query
  ↓
再建立 ReadingPlan 和证据需求
  ↓
再改结构化切分和关系边
  ↓
再做多视图召回和 Evidence Packet
  ↓
最后评估是否需要完整森林优化
```

最值得直接应用到当前 Agent 的不是某个特定公式，而是四条设计原则：

1. **问题先于检索**：模糊 prompt 先变成可验证子问题；
2. **结构先于相似度**：相似度找入口，结构关系补上下文；
3. **增量价值不同于直接相关**：条件、限制和反证可能低相似但不可缺少；
4. **选择证据包而不是 Top-K chunk**：最终上下文必须覆盖回答该问题所需的证据类型。

沿着这条路径，Paper-Agent 可以从当前的 Chunk RAG 逐步升级为 Query-Planned Evidence Graph Reading，同时保留现有页码、证据 ID、来源版本、视觉哈希、范围校验和失败降级等可靠性基础。
