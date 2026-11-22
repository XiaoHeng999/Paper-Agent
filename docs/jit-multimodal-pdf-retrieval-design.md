# Query 驱动的 JIT 多模态 PDF 解析、检索与多跳证据设计

## 1. 文档目的

本文记录 Paper-Agent 对即时论文分析场景的目标方案，重点回答：

1. 同一 PDF 页面同时包含文本、表格、公式和图像时如何切分；
2. 文本、Caption、OCR、视觉对象和 Query 如何建立关联；
3. 哪些处理适合预先完成，哪些处理应在 Query 到来后 JIT 执行；
4. OCR 和图像 embedding 是否必要；
5. 该方案能够解决哪些语义问题；
6. 多跳语义和多跳证据链是否已经被解决；
7. 后续应如何安全实现有界多跳检索与链路核查。

相关文档：

- [PDF 阅读链路问题分析](./pdf-query-chunking-retrieval-analysis.md)
- [Query-Planned Evidence Graph 实施方案](./query-planned-evidence-graph-upgrade.md)
- [图像 Embedding 策略](./image-embedding-strategy.md)
- 参考思想：[GraphMemix](https://arxiv.org/pdf/2608.26983)

---

## 2. 核心结论

当前最适合 Paper-Agent 的方案是：

```text
低成本、Query 无关的对象发现和来源版本化
+ Query 到来后的问题拆解
+ 多视图检索
+ 有界关系扩展
+ 候选级 JIT OCR
+ 可选候选级 Image Embedding
+ VLM 原图理解
+ Evidence Packet 和逐 Claim 核查
```

不建议：

- 将整页直接作为一个统一 chunk；
- 在用户问题未知时对全文做昂贵生成式摘要；
- 对所有页面无条件 OCR；
- 对所有图片无条件生成 image embedding；
- 使用孤立 Top-K chunk 直接回答复杂问题；
- 沿语义相似边进行无界多跳扩展；
- 将 embedding 相似度视为事实支持。

这套方案改善了语义召回和证据完整性，但不能宣称已经完全解决多跳语义问题。它提供的是解决多跳问题所需要的表示、关系和控制框架。

---

## 3. 同页文本与图像如何拆解

### 3.1 页面不是最终检索单元

一个 PDF 页面可能同时包含：

- 双栏正文；
- Figure；
- Table；
- Equation；
- Caption；
- 页眉页脚；
- 图表讨论段落。

因此不能简单表示为：

```text
Page 5 = 一段文本 + 一张整页图片
```

目标表示应是：

```text
Page 5
├── TextBlock 1
├── Equation 3
├── TextBlock 2：公式解释
├── Figure 2
├── Figure 2 Caption
├── TextBlock 3：Figure 2讨论
├── Table 1
└── Footer
```

### 3.2 每个对象的基础字段

```json
{
  "id": "paper:page5:figure2",
  "paper_id": "paper",
  "page": 5,
  "bbox": [100, 220, 480, 510],
  "object_type": "figure",
  "reading_order": 6,
  "section_path": ["4 Experiments", "4.2 Ablation"],
  "source_version": "...",
  "relations": []
}
```

解析器不能可靠获得 bbox 时，必须保存：

```json
{"bbox": null}
```

此时只能使用匹配来源版本的整页图像作为回退，不能声称完成了对象级裁剪。

### 3.3 文本切分

文本切分顺序：

```text
章节
→ 段落
→ 句子
→ 最大字符限制
```

每个 TextBlock 保留：

- page；
- section path；
- bbox；
- parent；
- previous/next；
- overlap；
- 引用的 Figure/Table/Equation ID。

文本块即使被长度限制拆分，也不能丢失：

- 页码；
- 章节；
- 引用关系；
- 原始父段；
- source version。

### 3.4 视觉对象切分

Figure、Table 和 Equation 不按字符切分，而是成为 canonical object：

```text
VisualObject Figure 2
├── image crop view
├── Caption view
├── OCR view
├── nearby discussion view
├── page image view
└── source/image version
```

不同 view 不能预先合并成一个长字符串。它们是同一对象的多个检索入口。

---

## 4. 预处理与 JIT 处理的边界

### 4.1 预先完成的低成本工作

论文进入系统后，可执行：

```text
PDF source hash
页面目录
原生文本提取
章节/段落识别
Figure/Table/Equation编号
Caption候选
原生布局和bbox（可用时）
确定性结构关系
```

这一阶段不应默认执行：

- 全文 OCR；
- 全量图像 embedding；
- 所有页面 VLM 理解；
- Query 无关的深度摘要；
- 大规模语义关系生成。

### 4.2 Query 到来后的 JIT 工作

```text
用户 Prompt
→ Reading Plan
→ ReadingSubquery
→ Evidence Requirements
→ 文本/结构检索
→ 发现缺口
→ 候选级 OCR / crop / image embedding / VLM
→ Evidence Packet
```

JIT 的对象是：

- Query embedding；
- 当前子问题相关的证据检索；
- 候选对象的 OCR；
- 候选 crop 的 image embedding；
- 候选关系扩展；
- Node/Increment 判断；
- Evidence Packet 组织。

不建议每个 Query 都重新解析整篇 PDF 或重新计算所有内容向量。

---

## 5. Embedding 如何使用

### 5.1 文本 Embedding

文本候选包括：

- 正文；
- Caption；
- OCR；
- 表头和结构化单元格；
- 公式附近解释；
- 附近讨论。

对于单篇或少量论文，可以在当前任务内批量生成向量并放入内存，不必使用长期向量数据库。

```text
Candidate text views
→ Task-local embeddings

ReadingSubquery
→ Query embedding

Cosine similarity
→ Text seeds
```

### 5.2 Image Embedding

Image embedding 只用于视觉对象召回：

```text
VisualObject crop
→ Image embedding

视觉型 Query
→ Text embedding in shared multimodal space

Similarity
→ Visual seeds
```

它不能替代：

- 图表数值读取；
- 表格行列恢复；
- 曲线和误差条判断；
- 实验条件核查；
- Claim 验证。

### 5.3 多视图融合

同一对象可以通过不同通道命中：

```text
Text dense
Lexical
Exact Figure/Table reference
Caption
OCR
Section
Requirement
Optional image embedding
```

初始阶段使用 RRF 或 max-over-views 均可，但必须保留每个命中的原始通道，不能只保存一个不可解释的总分。

---

## 6. OCR 是否必要

OCR 不是所有对象都必须执行，但也不能完全移除。

### 6.1 不需要 OCR

- PDF 有可靠原生文本；
- 表格单元格可直接解析；
- Caption 和正文足以定位视觉对象；
- 图中没有重要文字；
- Query 不依赖图内标签或精确字符串。

### 6.2 需要 OCR

- 扫描 PDF；
- 图片表格；
- 架构图模块名；
- 曲线图坐标轴、图例和单位；
- Query 包含 FP16、A100、batch size 等精确文字；
- 图内文字没有出现在正文或 Caption 中。

### 6.3 推荐方式

```text
先用原生文本、Caption和结构关系缩小候选
→ 对少量候选对象 JIT OCR
→ OCR结果作为一个view进入检索和Reader
→ 重要结论仍回看原图
```

OCR 解决“图中写了什么”，VLM 解决“图中的空间和视觉关系是什么”。两者不能互相完全替代。

---

## 7. Query 与切分对象如何直接关联

Query 与文档对象之间应存在多种关联函数：

```text
Query ↔ TextBlock：文本语义相似度
Query ↔ Caption：文本语义相似度
Query ↔ OCR：文本语义相似度
Query ↔ Image crop：图文相似度
Query ↔ Figure编号：精确匹配
Query ↔ Section：章节匹配
Query ↔ Evidence Requirement：类型匹配
```

融合后选择 Top-L seed，再通过关系扩展：

```text
caption_of
same_object
discusses
defines
same_page
previous/next
same_section
experiment_condition_of
```

因此 Query 不直接改变 PDF 的原始事实边界，而是决定：

- 哪些对象进入候选；
- 哪些对象需要更细粒度 JIT 解析；
- 沿哪些关系扩展；
- 哪些 Evidence Requirements 仍有缺口；
- 最终组装哪些上下文。

---

## 8. GraphMemix 思想如何迁移

GraphMemix 的核心过程是：

```text
Multi-view retrieval
→ Top-L seeds
→ bounded relation expansion
→ Node Verifier
→ Evidence-Chain Verifier
→ Evidence Forest selection
→ Reader
```

它的 JIT 重点是 Query 到来后重建 query-relevant evidence subgraph。基础多模态表示和 embedding 仍然存在，并非完全在 Query 后从零生成。

迁移到论文阅读后，Memory Unit 应替换为：

```text
TextBlock
Figure
Table
Equation
Caption
OCRView
VisualTranscription
ExperimentSetting
```

Schema Edge 应替换为：

```text
caption_of
discusses
defines
same_object
same_page
adjacent
same_section
condition_of
limitation_of
```

当前 Paper-Agent 已经实现了其中一部分：Reading Plan、多视图文本召回、Evidence Packet、`same_object`、邻接扩展，以及可选 Node/Increment Verifier。

---

## 9. 语义问题是否已经解决

### 9.1 已经明显改善的部分

#### Query 语义过载

原始 Prompt 被拆成 ReadingSubquery 和 Evidence Requirements，不再用一个长 topic 同时承担所有召回任务。

#### 多视图语义入口

同一 Figure/Table 可以通过 Caption、OCR、正文引用和图像 view 命中，降低单一表示漏检。

#### 低相似度互补证据

候选可以通过 `same_object`、邻接和父节点关系扩展，不完全依赖 Query 相似度。

#### 重复与互补区分

Node/Increment Verifier 可以区分：

```text
new_fact
clarification
corroboration
redundant
conflict
irrelevant
```

#### 证据完整性

Evidence Packet 按 requirement 覆盖选择，而不是只取最相似的若干 chunk。

### 9.2 尚未完全解决的部分

#### 隐式语义关系

论文可能不明确写出：

```text
这个近似假设限制了后续速度结果
```

系统需要理解“假设—方法—实验结果”的隐式约束关系。单纯 embedding 和邻接边不能可靠解决。

#### 指代消解

例如：

```text
this module
the former setting
our second variant
该机制
上述结果
```

这些指代可能跨段、跨页或跨 Figure，需要专门的 coreference 和实体对齐。

#### 同义但不同条件

两个段落都谈“latency improvement”，但可能分别基于：

- 不同硬件；
- 不同精度；
- 不同 batch size；
- 不同数据集。

语义相似不能表示实验条件一致。

#### 反事实与否定

Embedding 往往无法稳定区分：

```text
方法提高性能
方法没有提高性能
只在短序列上提高性能
```

仍需要 Query-conditioned verifier。

所以更准确的结论是：

> 当前方案解决了语义检索入口和证据组织问题，但没有完全解决深层语义判断。

---

## 10. 多跳语义是否已经解决

答案是：没有完全解决，只建立了多跳所需的基础图和受控扩展机制。

### 10.1 三类“跳”必须区分

#### A. 确定性结构跳

```text
正文引用 Figure 2
→ Figure 2
→ Figure 2 Caption
→ Figure 2 OCR
```

关系来自编号、页码和对象身份，可靠性较高。当前方案已经能较好处理。

#### B. 指代与文档结构跳

```text
“该模块”
→ 前文定义的 Router
→ Router 的公式
→ Router 的消融实验
```

部分关系可以通过章节、邻接、实体名和引用恢复，但仍需要 coreference 和实体规范化。

#### C. 推理语义跳

```text
速度结果
→ 实验设置使用 FP16
→ 基线精度未报告
→ 公平性无法确认
```

这不是简单的图遍历，而是需要模型判断：

- 第二条证据是否真正限定第一条；
- 条件是否属于同一个实验；
- 缺失信息是否足以影响结论；
- 最终只能得出 qualified 还是 contradicted。

这类多跳必须通过链路核查，不能只靠 embedding 或图边。

### 10.2 当前已经具备的多跳基础

- `previous/next`；
- `same_parent`；
- `same_object`；
- Caption/OCR/讨论多 view；
- query-aware seed；
- 一跳有界扩展；
- Node direct support；
- Anchor-conditioned Increment role；
- Evidence Requirement gap；
- Claim-level Verifier。

### 10.3 当前仍缺少

- 可靠的 `defines`、`condition_of`、`limitation_of`；
- 跨页实体和指代消解；
- 实验 ID 与配置对齐；
- 关系边置信度和来源；
- 完整 Evidence Chain Verifier；
- 对整条链而不是单个 Anchor pair 的一致性判断；
- 多跳失败后的定向修复；
- 多跳召回的离线评测集。

---

## 11. 推荐的有界多跳方案

### 11.1 不允许无界语义遍历

禁止：

```text
Query seed
→ 相似节点
→ 更多相似节点
→ 更多相似节点
→ 任意扩张
```

这会导致：

- 主题漂移；
- 重复证据；
- 错误边累积；
- 上下文爆炸；
- 不相关段落污染推理。

### 11.2 使用分层边白名单

第一跳优先可靠关系：

```text
caption_of
same_object
explicit_reference
previous/next
definition_of
```

第二跳只在 requirement 仍有缺口时允许：

```text
condition_of
experiment_setting_of
ablation_of
limitation_of
```

不应默认允许第三跳。确有需要时，必须由明确 gap query 驱动。

### 11.3 每条边保存依据

```json
{
  "source_id": "result-span",
  "target_id": "setup-span",
  "relation": "condition_of",
  "confidence": 0.82,
  "provenance": "explicit experiment id + shared table reference",
  "query_scope": "latency fairness"
}
```

没有 provenance 的模型推断边不能与编号、页码形成的确定性边同等对待。

### 11.4 使用 Evidence Requirement 控制扩展

例如：

```text
当前已覆盖：result, baseline
仍缺：hardware, precision, limitation
```

下一跳只能寻找：

- 实验设置；
- 表注；
- 附录配置；
- 限制和失败场景。

不能继续召回更多重复结果段落。

### 11.5 对链路整体核查

候选链：

```text
Claim：速度提高30%
  ↓ supported_by
Table 2
  ↓ condition_of
实验使用FP16
  ↓ comparison_gap
基线精度未报告
```

Chain Verifier 应判断：

1. 是否属于同一个实验；
2. Table 2 是否真的报告30%；
3. FP16 是否适用于该结果；
4. 基线精度是否确实未报告，而不是尚未检索到；
5. 最终结论应是 supported、qualified、contradicted 还是 unknown。

---

## 12. 多跳示例

用户问题：

> 论文声称推理更快，这个优势是否来自更低精度？

### 第一步：问题规划

```text
result
baseline
hardware
precision
budget
table
limitation
```

### 第二步：直接召回

```text
结果段落：latency降低30%
Table 2 Caption
```

### 第三步：结构扩展

```text
Table 2 Caption
→ same_object
→ Table 2结构化单元格/OCR/原图
```

### 第四步：Requirement Gap

```text
已覆盖：result, baseline, table
缺少：hardware, precision, budget, limitation
```

### 第五步：第二跳

```text
Table 2
→ experiment_setting_of
→ 实验设置段落

实验设置
→ appendix_configuration
→ 附录实现配置
```

### 第六步：链路核查

```text
结果使用FP16
基线精度unknown
硬件相同
batch size相同
```

最终不是直接输出“不公平”，而是：

```text
速度结果得到论文内部证据支持；但基线精度未在当前已审证据中确认，因此无法判断该优势是否完全来自方法本身，结论应标记为 qualified。
```

---

## 13. 当前实现状态

### 本轮落地范围

Evidence Packet 现在为关系扩展命中的候选记录一条实际检索路径。每跳包含源和目标 evidence ID、关系、confidence 与 provenance。第一跳来自邻接、父段或对象 ID。对已选第一跳候选，若证据包仍缺硬件、数值精度、预算、实现、消融或限制，可按正文中共同出现的规范化图表或公式编号再找最多四个候选，然后按原预算重新组包。这里的 confidence 描述结构匹配，对实验条件是否一致仍需读原文。

单篇 Claim 同时引用第二跳路径两端时，Verifier 在上下文预算内读取中间节点，判断方法、实验与配置是否一致。缺失链路判断或配置无法确认时，原本可通过的结论降级为 qualified。已记录的缺口表示当前证据范围内未确认。同页存在多个视觉对象时，无明确编号的整页转录保留为页级证据。

这一实现覆盖显式编号路径。隐式实验对齐、对象裁剪、候选级 OCR 和图像 embedding 仍按下列阶段推进。

### 已实现

- ReadingSubquery 和 Evidence Requirements；
- 结构感知文本切分；
- 文本 dense、lexical、section、exact reference、requirement 召回；
- RRF 融合；
- `previous/next`、`same_parent`、`same_object`；
- Figure/Table/Equation canonical object；
- Caption、OCR、讨论多 view；
- Evidence Packet；
- gap query；
- 可选 listwise Node/Increment Verifier；
- Claim-level Reader 和 Verifier；
- source/image version 核查。

### 部分实现

- 多视图视觉对象检索；
- 一跳关系扩展；
- requirement 覆盖选择；
- 扫描页视觉转录；
- 整页原图核查。

### 尚未实现

- 可靠 layout/bbox parser；
- Figure/Table 对象级 crop；
- 结构化 Table cell graph；
- 候选级 JIT OCR 调度器；
- 临时 image embedding；
- `condition_of`、`limitation_of` 等语义边；
- 实体、指代和实验配置对齐；
- 完整多跳 Evidence Chain Verifier；
- 多跳召回评测集。

---

## 14. 后续实施顺序

### Phase A：Layout 与对象边界

1. layout-aware parser；
2. reading order；
3. Figure/Table/Equation bbox；
4. Caption 与对象绑定；
5. 对象 crop 和版本。

### Phase B：JIT 感知

1. 基于 requirement gap 选择视觉对象；
2. 候选级 OCR；
3. Table 结构解析；
4. VLM 只读取 Top-K crop；
5. 失败时整页回退。

### Phase C：语义关系

1. 实体规范化；
2. Figure/Table/Equation 引用解析；
3. 实验 ID 与配置对象；
4. `condition_of`、`ablation_of`、`limitation_of`；
5. 每条关系记录 confidence 和 provenance。

### Phase D：有界多跳

1. 一跳确定性扩展；
2. requirement 驱动的第二跳；
3. Evidence Chain Verifier；
4. 链路冲突和条件一致性判断；
5. 最多一轮 gap-driven 修复。

未来需要实现多跳逻辑证据检索：以用户子问题和未满足的 Evidence Requirement 为起点，先沿 `caption_of`、`same_object`、`defines` 等高置信度关系找到直接证据，再按缺口有界扩展到 `condition_of`、`result_of`、`ablation_of` 和 `limitation_of`。每一跳必须保存 relation、confidence、provenance 和对应 evidence ID，最终由 Evidence Chain Verifier 检查各节点是否属于同一方法、实验和配置；任何一步无法确认时，结论必须降级为 `qualified` 或 `unknown`，不能依靠更多跳数补造缺失事实。

### Phase E：可选 Image Embedding

1. 多模态 provider；
2. 任务内存索引；
3. 只在视觉 requirement 缺口时运行；
4. 与 Caption/OCR/结构召回融合；
5. 离线 A/B 评测。

---

## 15. 验收标准

### 同页混合内容

- 正文、Caption、Figure、Table 和 Equation 不被混成单一 chunk；
- 每个对象保留 page、bbox、reading order 和 source version；
- bbox 不可用时明确降级。

### 语义召回

- 同一对象可通过 Caption、OCR、正文引用或图像 view 命中；
- 低相似度条件证据可通过可靠关系进入候选；
- 重复结果段落不会挤占条件和限制证据。

### 多跳

- 每一跳都有 relation type、confidence 和 provenance；
- 默认最多一跳，第二跳由 requirement gap 触发；
- 不执行无界语义遍历；
- 最终结论能够展示完整证据链；
- 链路任何一步不确定时，结论降级为 qualified 或 unknown；
- “未检索到”不能写成“论文没有报告”。

### 成本

- 不全量 OCR；
- 不全量生成 image embedding；
- VLM 只处理 Top-K 对象；
- 每个 Query 记录 OCR、embedding、VLM 和 verifier 调用成本；
- 任务结束可释放临时视觉索引。

---

## 16. 最终判断

这套方案已经为语义检索和多跳证据组织建立了合理基础，但不能把“存在图结构”误认为“已经解决多跳推理”。

当前最可靠的是：

```text
Query拆解
+ 多视图seed
+ 确定性结构扩展
+ requirement覆盖
+ Anchor-conditioned增量判断
+ Claim核查
```

下一步真正决定多跳质量的是：

```text
可靠语义边
+ 实体/实验配置对齐
+ 边的provenance
+ requirement驱动的第二跳
+ Evidence Chain整体核查
```

因此应将系统目标表述为：

> 以 Query 为中心，在严格预算和可追溯关系约束下，JIT 重建足以回答问题的多模态证据链；而不是通过更多 embedding 或更多跳数来假设语义问题自然消失。
