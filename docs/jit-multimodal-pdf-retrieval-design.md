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

这条多跳路径覆盖显式编号引用。视觉读取另外支持原生栅格图表的候选级裁剪与转录：

- PDFium 将带编号的 Caption 与相邻的顶层栅格对象或包含图形的 Form XObject 配对。使用同次渲染的坐标变换裁剪旋转页；唯一匹配原生文字的单页证据保留 PDF bbox、源版本和文字流顺序。只有位置与编号能一一对应时才保存 bbox，坐标顺序为 PDF 原生坐标系下的 left、bottom、right、top。独立绘制的矢量图可以通过包围多个图形元素的矩形外框定位；框外紧邻文字时回退。对图注与图内字母一一对应的连通面板网格，保存整图与每个面板的边界，支持 Figure 1(b) 定位。无法唯一配对时使用原页。
- 用户明确引用图表编号时，从已选页按 bbox 裁剪，每篇每次最多处理两个匹配对象。转录记录源 PDF、整页和 crop 的版本，作为该对象的 OCR view 参与检索。核查阶段仍读取匹配版本的原页以检查上下文。
- 通用摘要问题不触发对象裁剪转录。当用户询问数值精度或硬件而未指定图号时，先用原生文本建立用户子问题的证据包；缺口存在时用图注主题和已召回的正文引用选择裁剪。显式编号与缺口调度共享每篇两个对象的预算。最终检索会纳入转录，并分别记录精度与基线精度需求。对相关的系统结果推论，Verifier 必须定位同一实验双方的精度来源，否则限制归因结论。

明确表号的正文或裁剪转录中，若存在单个矩形管道表格或显式 HTML 分组表头，系统会为每个非空数据单元格保存证据 ID、行列位置、行头、列头、原值及父证据。保留单位，并绑定本段中标记唯一的脚注。相邻页同表号且明确标记续表时，可从同一源版本继承列头并延续行号。核查续页单元格时会一并读取表头证据及相应原页。列数不齐或来源不明确时保留原证据。

对以“该模块”“Under this setting”等开头的被引段落，Verifier 在预算内从同篇同章节的前两个邻接片段查找所指对象，返回候选 ID、所指名称和逐字引句。无法确认时保留限定，越界或虚构引句会导致核查失败。

对正文中全称首字母与括号缩写一致、且同篇只有一个全称的实体，第一跳可沿 `definition_of` 跨页找回定义；同缩写有多个全称时跳过。对于带显式实验编号的设置、消融和局限，第二跳也可以用同一编号和目标段落的直接标记建立关系。如果两端都给出实验编号但编号不同或不唯一，共享图表编号也不会触发这条扩展。对系统推论，如果已选第二跳终点被对应子问题的 Claim 引用，核查时会补读路径起点和中间节点；单篇卡片列出边、依据及链路判断。受控样例可运行 `python -m evaluation.multihop_retrieval` 检查正确路径和误连。

图文检索默认使用本地 `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`（revision `9cfa6413f704a7c1cf5064d240748e10c876b286`）。模型按需加载，将问题与「原图裁剪＋对应图注」分别编码；缺少图注时使用裁剪本身。每篇最多比较八个可定位对象，前两个结果参与证据选择。权重在同一进程内复用，裁剪向量按任务、图像版本、坐标和图注缓存。首次加载时校验本地权重、推理代码和配置的 SHA-256。缺少依赖或模型文件时记录警告并继续文本检索。

阅读任务的「图文检索模型」提供 GME、CLIP 和关闭三个选项。API 可设置 `image_embedding_model` 为 `qwen2-vl` 或 `clip`，也可用 `image_embedding_enabled: false` 关闭。GME 依赖安装命令为 `uv sync --locked --extra gme`，默认权重目录为 `/data/models/liveact/qwen-vl/gme-Qwen2-VL-2B-Instruct`，可用 `PAPER_AGENT_GME_DIR` 指定。CLIP 使用 `Xenova/clip-vit-base-patch32` 的本地量化 ONNX 文件，revision 为 `6b7c523703ae26e39be4b03774966f8571c7f523`，默认从 `$XDG_CACHE_HOME/paper-agent/clip-vit-base-patch32/<revision>` 加载（未设置 XDG_CACHE_HOME 时使用 ~/.cache），支持 `PAPER_AGENT_CLIP_DIR`。

使用 `uv run --locked --extra gme --extra image-retrieval python -m evaluation.visual_retrieval <PDF路径> --full-text --read-crops --image-model qwen2-vl` 可以从原始 PDF 复现全文候选、对象定位、原页版本和图内读字核对。将模型参数改为 `clip` 可运行同题对照。

六篇 PDF、十五个定位问题上，纯文本首选命中 5/15，CLIP 命中 12/15，GME 联合图注命中 13/15。十三个 GME 命中均能回查正确版本的原页。GME 找回了 CLIP 漏掉的 Transformer 架构图和 ResNet 网络层图，但漏掉了 CLIP 找对的 ResNet 误差曲线图；两者都漏掉 DETR 流程图。十三个带可见文字标注的问题中，两种模型各有十题同时通过对象、原页及原生文字或本地 OCR 匹配。其中一篇论文只有一个可裁剪候选。该组数字衡量现有开发样本上的整条检索路径，不能区分编码器与附加图注各自的贡献。

在三篇缓存论文上运行 `python -m evaluation.pdf_evidence_pairs data/paper_cache` 可重建全文、校验源文件哈希并按原页逐段核对。三个问题都要求图注及另一页的正文同时进入同一证据包，当前命中 3/3。受控两跳评测另有七个正例和两个负例，检查关系类型以及是否出现误连。两组定向评测的标注由 AI 准备。

独立人工证据基准使用 Dasigi 等人发布的 [QASPER v0.3](https://huggingface.co/datasets/allenai/qasper)。将[原始压缩包](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz)传给 `python -m evaluation.qasper_evidence <压缩包>` 即可复现。程序校验 SHA-256，只读取 dev split。1005 个问题中，34 个问题（31 篇论文）满足筛选条件：单一用户子问题、每个人工答案的证据均可唯一定位、每组证据跨至少两个章节。以完整论文段落为候选，不把标注输入检索器。默认五个节点和 4200 字符预算下，独立融合用户问题的 BM25 排序后，完整命中任一标注证据组从 1/34 增至 2/34，平均段落召回率从 16.7% 增至 20.1%。该指标衡量跨章节证据覆盖，标注不包含有向推理边。

本地生成评测在上述三个真实 PDF 的证据包上运行 Qwen3-0.6B（revision `c1899de289a04d12100db370d81485cdf75e47ca`，贪心解码）。在双段评测命令后添加 `--reader-inputs <新文件>`，然后在安装 PyTorch 和 Transformers 的环境运行 `python -m evaluation.local_pdf_answers <新文件> <本地模型目录>`。模型只看到问题和编号原文。两题覆盖全部预设事实词组，一题同时引用两段目标证据，没有一题同时满足这两个条件。事实词匹配无法检验答案中的所有陈述。

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

### 已验证的处理范围

- **跨页表格**。相邻页的原文片段必须双向相连，且来源版本相同。无续表标题时，重复列头、连续数值行号或相邻页边缘对齐的原生坐标可提供拼接依据。单元格记录前页证据 ID，核查时读取相应原文。HTML 表体的 rowspan、colspan 保存为一个带范围的值，非法重叠和越界会放弃结构化。带标记的脚注和明确写有“所有数值”的本段表注可绑定到单元格。
- **对象边界**。除带框图外，系统支持由横纵轴及相连绘图元素构成、外围留白的矢量图。带图内字母的相邻面板可以错位。图注明确说明两个面板且两图之间留白时，可生成整图裁剪。独立面板的标签必须有自己的定位证据。
- **按需感知**。用户明确询问指标、消融、计算预算、精度、硬件或图中文字时，先检查文本证据包。缺少对应需求且存在相关图注或编号引用才进入裁剪候选，共享原有对象预算。
- **跨章节指代与多跳核查**。含指代表达的结果段可在同版论文的前页查找设置段。跨章节候选至少共享两个有区分度的命名标识；多个匹配配置保持待定。实验编号与原文条件、结果、消融和局限标记可以产生对应的有类型关系。两跳链路判断一致时须给出三个节点各自的逐字原文。引句缺失会降级，虚构的引句会使核查失败。

### 功能基线与质量判断

A–E 五个阶段均已有运行入口。对于“图 3 的加速结果是在什么硬件和精度下取得的”这类问题，系统可以按图号定位结果，查找实验设置，将原文放入证据包，再核查答案中的引用。当前版本可作为定向论文阅读的功能基线。复杂问题的质量取决于三个连续判断：候选中能否找齐证据，有限预算下能否把互补段落一起保留，答案能否把每个判断指回支持它的原文。

QASPER 跨章节评测的 34 道题，可以按完整证据组最先丢失的位置逐题归类。这里的“完整”表示覆盖至少一位标注者给出的整组原文，预算为每题最多 5 段、4200 字符。

| 首次出现问题的环节 | 题数 | 对改进的含义 |
|---|---:|---|
| 候选中找不齐完整证据组 | 13 | 应检查召回渠道能否找到分散在不同章节的原文 |
| 候选中有完整组，但其中没有一组能放进默认预算 | 1 | 应检查段落粒度和必要上下文的长度 |
| 候选中有完整组且放得下，最终选段时出现遗漏 | 18 | 应检查打分和停止条件能否保留回答同一问题所需的互补段落 |
| 最终证据包包含完整组 | 2 | 可继续检查生成答案是否引用正确原文 |

把字符预算加倍到 8400，完整命中从 2/34 增至 3/34；只把段数上限从 5 提到 8，仍为 2/34。对当前离线配置，优先改选段有直接的实验依据。例如，“提升了多少”和“在哪种配置下成立”各需要一段原文；一段结果同时命中多个需求类别，也可能没有交代该实验的配置。[证据选择实现](../src/research/selection.py) · [人工证据评测](../evaluation/qasper_evidence.py)

### 改进顺序与验证方法

1. **优先保住已经找到的证据。** 固定候选、5 段上限和 4200 字符预算，对照现有选段策略与按问题分别保留结果、条件、局限的策略。先复查上述 18 道，再用全部 34 道统计新增完整命中和原有命中损失，避免只修复已知错题。人工证据只用于评分。
2. **对候选缺失的题单独改召回。** 针对 13 道找不齐的题，在相同候选数下对照 BM25、文本向量及二者融合。分别记录完整组进入候选和进入最终证据包的比例，以判断收益在哪一步保留下来。
3. **单独验收答案引用。** 固定已经找齐的证据包，逐项核查答案中的数值、配置和限制是否由所引段落支持。分别记录“答到要点但引错”和“引对原文但漏答”。已有三题本地生成测试足以暴露两类错误，估计发生率还需要更多独立样本。
4. **在视觉排序和读字环节按需增加计算。** GME 的两道定位错题中，目标均排第二；可让系统对前两张图逐项核对问题描述的布局和数量。对已经定位但读不清的小字，另测文本框级 OCR。按论文留出新样本，分别统计复核纠正数、新增错误、正确定位后的读字率和每题延迟，再决定是否调整默认策略。[视觉错题与消融实验](jit-visual-retrieval-diagnosis.md)

### 验收边界

缺少可辨边界的图、排版不明确的续表及无法确定作用域的脚注保留原页或原段。没有共同实验标识、存在多个竞争配置的跨章节归因保留为待确认。真实论文的跨页成对召回和短文字读取已有离线评测。通用跨章节召回和生成答案的引用完整性仍有实测缺口。QASPER 提供人工段落标注，本项目的图像问答尚未获得独立专家逐题评分。配置的线上模型服务在四项联测中返回 HTTP 402（余额不足）。


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
