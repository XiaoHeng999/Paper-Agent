# Paper-Agent 图像 Embedding 策略：文本优先、临时内存与持久化索引

## 1. 决策结论

Paper-Agent 当前是以“输入论文并即时分析”为主的系统，不应把图像 embedding 与长期向量数据库绑定，也不应把它作为论文阅读的强制前置步骤。

建议同时保留三种能力边界：

1. `text_structure`：不生成图像 embedding，使用 Caption、OCR、正文引用和对象关系检索；
2. `ephemeral_image`：为本次任务按需生成对象级图像 embedding，只保存在运行时内存或短期可删除缓存中；
3. `persistent_image`：面向长期论文库建立持久化图像向量索引。

当前推荐：

> 方式一作为默认主路径；方式二作为可选的召回补救和实验能力；方式三暂不实现。

方式二是三者中最适合即时论文分析的图像 embedding 方案，但“适合保留”不等于“现在必须默认启用”。应先完成可靠对象裁剪和离线漏检评测，再决定它是否进入默认流程。

---

## 2. 三种方式分别解决什么问题

### 2.1 方式一：不使用图像 Embedding

流程：

```text
用户子问题
  ↓
Caption / OCR / 表头 / 正文引用 / Figure 编号检索
  ↓
沿 same_object 关系组装视觉对象的多种 view
  ↓
按需加载原图或整页
  ↓
视觉模型核查
```

依赖的检索信号：

- Figure、Table、Equation 编号；
- Caption；
- OCR 或视觉转录；
- 表头、坐标轴标签和单位；
- 附近正文；
- “as shown in Figure 3”一类引用关系；
- 章节路径、页码和对象类型；
- `same_object`、`adjacent` 等确定性关系。

优点：

- 不需要新的 embedding 模型或存储服务；
- 结果容易解释和回查；
- 对论文图表中的数字、标签和编号通常更有效；
- 不会因为通用视觉模型只识别出“折线图”而忽略具体实验含义；
- 当前 Paper-Agent 已经具备大部分基础能力。

缺点：

- Caption 和 OCR 很弱时可能漏检；
- 视觉模式本身与 query 相关、但文字没有明确描述时召回不足；
- 架构图、示意图和无文字曲线图较依赖附近正文。

适用场景：

- 普通论文即时阅读；
- 图表有 Caption、编号或附近讨论；
- 用户询问具体实验、指标、表格数值或某个 Figure/Table。

### 2.2 方式二：运行时临时图像 Embedding

流程：

```text
文本和结构检索无法充分覆盖子问题
  ↓
选择少量 Figure/Table 对象
  ↓
按 bbox 渲染对象 crop
  ↓
在图文共享向量空间生成 image embedding
  ↓
在本次任务内做内存相似度检索
  ↓
命中的对象仍需 Caption/OCR/原图核查
  ↓
任务结束释放向量
```

这里不需要 Chroma、Milvus 或 Qdrant。假设一次任务有 10–100 个视觉对象，可以直接保存为内存数组并计算余弦相似度：

```python
scores = image_vectors @ normalized_query_vector
top_indices = scores.argsort()[-k:][::-1]
```

优点：

- 能补充 Caption/OCR 召回不到的视觉语义；
- 不建设长期知识库；
- 不产生跨用户或跨任务的数据混合；
- 索引规模很小，检索实现简单；
- 可以只对文本检索不确定的对象执行，成本可控。

缺点：

- 必须使用真正支持图文共享空间的 embedding 模型；普通文本 embedding API 不能直接替代；
- 需要可靠的对象 bbox 和 crop，整页 embedding 容易被正文、页眉和多个图表干扰；
- 对图表数值、单位、图例和细线差异仍可能不敏感；
- 每次任务计算向量会增加延迟和费用；
- 图像向量只能用于候选召回，不能成为最终证据。

适用场景：

- 架构图、流程图或视觉示意图缺少有效 OCR；
- 用户用视觉语义描述目标，而不是给出 Figure 编号；
- 一篇或少量论文内进行对象搜索；
- 不需要长期跨论文复用。

### 2.3 方式三：持久化图像向量数据库

流程：

```text
论文持续进入共享语料库
  ↓
抽取并版本化视觉对象
  ↓
批量生成图像 embedding
  ↓
写入持久化向量数据库
  ↓
跨会话、跨论文、跨用户检索
```

只有以下目标成立时才值得考虑：

- 维护数千篇以上的长期论文库；
- 同一论文和图表会被反复查询；
- 需要跨论文搜索相似架构图、实验曲线或系统设计；
- embedding 生成成本需要跨会话复用；
- 有清晰的权限隔离、删除、更新和模型迁移要求。

它带来的额外问题包括：

- 用户、项目和租户隔离；
- PDF 删除后的索引级联删除；
- source hash、crop hash 和模型版本失效；
- embedding 模型升级后的全量迁移；
- 视觉对象重复、论文版本重复和 canonical identity；
- 长期存储成本与数据治理。

因此，方式三属于“论文知识库”能力，不是当前即时阅读 Agent 的必要组成。

---

## 3. 为什么当前不能直接用整页图像 Embedding 代替对象裁剪

一页论文可能同时包含：

- 两张 Figure；
- 一个 Table；
- 正文段落；
- 页眉页脚；
- 多栏排版；
- 多个公式。

对整页生成单个向量会把这些语义压缩到一起。即使 query 只关注 Figure 3，页内其他内容也可能主导向量。

因此方式二的正确检索单元应当是：

```text
VisualObject crop
+ Caption view
+ OCR view
+ nearby discussion view
+ page/version metadata
```

当前系统已经建立了 `VisualObject` 和多 view 关系，但在解析器不能可靠给出 bbox 时明确保存 `bbox=null`。在这个阶段直接对整页做 image embedding，只适合作为实验性 fallback，不能声称已经实现对象级视觉检索。

---

## 4. 是否需要实现方式二

### 4.1 判断

建议实现，但应采用以下定位：

> 方式二是可选的、按缺口触发的视觉召回补救，不是默认必经步骤。

原因：

1. 它符合即时论文分析，不要求长期数据库；
2. 当前已有 ReadingSubquery、Evidence Requirement、VisualObject 和 gap retrieval，可自然判断何时触发；
3. 对无有效文字的架构图和示意图可能有真实收益；
4. 内存检索的工程复杂度不高；
5. 但当前缺少可靠 bbox 和图文 embedding provider，立即默认启用会得到粗糙的整页向量。

### 4.2 不建议立即默认启用的原因

- 当前 provider 接口主要是文本 embedding，尚未证明支持图像输入；
- `bbox=null` 时只能嵌入整页，候选粒度不够准确；
- 没有视觉漏检评测集，无法判断收益是否超过额外延迟；
- OCR、Caption 和正文关系已经能解决大量图表定位问题；
- 视觉 embedding 的相似度不能替代原图核查。

所以“需要实现”的准确含义是：建立可插拔接口并做小规模实验，而不是马上部署为默认主链。

---

## 5. 推荐的触发策略

方式二不应对所有页面和对象无条件执行。推荐按以下条件触发：

```text
子问题包含 figure/table/architecture/curve/diagram 等视觉意图
或 evidence requirement 包含 figure/table
  ↓
方式一召回后仍缺关键 requirement
  ↓
存在 bbox 可用的视觉对象
  ↓
对未充分覆盖的对象生成临时 embedding
```

可以采用三档模式：

```text
image_retrieval_mode = "text_structure"   # 默认
image_retrieval_mode = "ephemeral"        # 显式启用方式二
image_retrieval_mode = "persistent"       # 未来知识库模式
```

`ephemeral` 模式仍应设置硬预算：

- 每篇最多处理的视觉对象数；
- 每次任务最多生成的图像向量数；
- crop 最大像素；
- query 数量；
- 最大耗时和费用；
- 只允许一轮视觉 gap retrieval。

---

## 6. 方式二的建议实现结构

### 6.1 Provider 接口

不能复用只接受字符串的文本 embedding 接口冒充图像 embedding。建议单独定义：

```python
class MultimodalEmbeddingProvider:
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        ...

    async def embed_images(self, images: list[bytes]) -> list[list[float]]:
        ...
```

必须保证 `embed_text` 和 `embed_images` 输出位于同一个向量空间。

### 6.2 临时索引

```python
class EphemeralImageIndex:
    object_ids: list[str]
    vectors: Matrix
    source_versions: list[str]
    crop_versions: list[str]
```

索引只属于当前研究任务。任务结束后释放，不写入现有 EvidenceIndex 的持久化 collection。

### 6.3 版本约束

每个临时向量至少绑定：

- `paper_id`；
- PDF `source_hash`；
- `visual_object_id`；
- `page`；
- `bbox`；
- `image_version`；
- `crop_hash`；
- embedding model 和 dimensions。

任何一项变化都不能复用旧向量。

### 6.4 与 Evidence Packet 的关系

图像 embedding 只增加一个召回通道：

```text
retrieval_channels += ["image_embedding"]
```

命中后应回到 canonical `VisualObject`，再通过 `same_object` 加载：

- Caption；
- OCR；
- 附近正文；
- 原图 crop；
- 必要时整页上下文。

只有原图核查和文本证据共同支持时，相关 claim 才能进入正常验证流程。

---

## 7. 何时才升级到方式三

只有同时满足以下条件，才应从方式二升级为持久化索引：

1. 同一图像 embedding 的跨任务复用率明显较高；
2. 临时生成 embedding 已成为主要延迟或费用来源；
3. 产品目标明确包含跨论文视觉搜索；
4. 已有稳定的 canonical visual object 和版本策略；
5. 已完成用户数据隔离、删除和迁移设计；
6. 离线评测证明视觉向量带来稳定增益。

如果这些条件不成立，持久化数据库只会增加状态管理和数据治理负担。

---

## 8. 建议的实施顺序

### Step 1：先建立评测集

收集三类样本：

- Caption/OCR 足以命中的图表；
- 只有视觉模式能够命中的对象；
- 图像 embedding 容易误召回的相似图表。

指标至少包括：

- Visual Object Recall@K；
- Evidence Recall@Budget；
- 每篇新增延迟；
- 每篇新增费用；
- 误召回率；
- 最终 claim 是否真正引用并核查原图。

### Step 2：完成 bbox 与对象 crop

没有可靠 bbox 时，不应把整页 embedding 的结果当作对象级检索结果。bbox 获取失败必须保持显式降级。

### Step 3：定义独立多模态 Provider

验证模型是否真正支持 text-to-image 相似度，并记录模型版本和维度。

### Step 4：实现内存索引

只针对一个任务中的少量对象，不引入外部向量数据库。

### Step 5：接入 gap retrieval

仅在文本/结构 Packet 仍有视觉需求缺口时运行一轮。

### Step 6：A/B 评测后决定默认策略

只有方式二在视觉对象召回率上有稳定收益，并且延迟和费用可接受，才考虑从显式开关变为自动 fallback。

---

## 9. 最终建议

对当前 Paper-Agent：

- 保留三种方式的架构边界；
- 默认继续使用方式一；
- 将方式二作为下一项可实验能力；
- 方式二使用任务内存索引，不使用长期向量数据库；
- 先完成 bbox/crop 和多模态 provider，再接入图像向量；
- 图像向量只负责召回，不负责事实判断；
- 暂不实现方式三。

因此，如果问题是“方式二是否值得实现”，答案是肯定的；如果问题是“方式二是否已经必要到应立即成为默认路径”，答案是否定的。最合理的选择是先实现受控、可关闭、按缺口触发的 MVP，并用评测结果决定它是否长期保留。
