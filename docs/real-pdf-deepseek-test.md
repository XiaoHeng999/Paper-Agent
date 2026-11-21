# Sol-Attn 真实 PDF / DeepSeek 测试

以下相对数据文件路径均位于 `data/paper/2607.24027/`。该目录被 Git 忽略，原始 PDF 与真实运行产物保存在本地；本文保留测试发现与复现命令。

来源：https://arxiv.org/pdf/2607.24027 。论文 PDF 为 `2607.24027.pdf`，16 页、39,798,695 字节；SHA-256 见 `download.json`。

## 配置和真实请求

使用 DeepSeek 官方 OpenAI 兼容接口 `https://api.deepseek.com`，模型为 `deepseek-flash`。本地 `config/model.json` 使用 `api_key_env: DEEPSEEK_API_KEY` 引用 `.env`，不写入密钥；这些文件均被 Git 忽略。运行时显式使用 uv 的 `--env-file .env` 加载凭据。

模型目录请求成功。通过项目实际 provider 分别完成文本 JSON 请求、PDF 首页 PNG 图像请求，结果见 `connection-test.json`；图像响应识别了论文标题和首页示例图。此连通性检查不是论文正确性评测。

## PDF 准备检查

- 解析出 59,429 字符、60 个全文证据片段。
- 13 项检查通过，详见 `preflight-results.json`。
- 分块按页保留了提取后的文字（忽略空白比较）；这不能证明 PDF 文本提取本身无识别错误。
- 预检渲染页为 12、13、1、5、3、6；实际研究运行的选页以报告 `visual` 字段为准。
- 页图见 `visual/`，文字与分块见 `paper.md`、`chunk.json`、`evidence.json`。

## 首轮暴露的问题与修复

首次真实分析在 600 秒预算到期时，`TimeoutError()` 的消息为空。核验异常处理直接将空消息写进要求非空的 `VerificationDraft.reason`，导致二次校验异常，报告未保存。日志保留为 `live-test-first-failed.log`，不能把这一轮说成成功。

修复后，空超时消息有明确兜底理由，结论保留为 `insufficient_evidence`，标记语义核查未完成；回归测试确认可以持久化报告。另修复 SDK 客户端的异步 `close()` 没有被 await 的问题，真实 SDK 关闭验证通过。

修复后的回归测试：151 项，147 通过、4 跳过；`uv pip check` 通过。自动化回归使用替身验证程序行为，`data/paper/2607.24027/` 中的 live 任务实际调用 DeepSeek。

## 复现

在项目根目录执行（`--live` 会产生真实 API 调用）：

```bash
uv run --locked python data/paper/2607.24027/run_case.py
uv run --env-file .env --locked python data/paper/2607.24027/run_case.py --live
uv run --locked python data/paper/2607.24027/validate_report.py
```

采用生产 `run_research_node`，单篇报告、跨篇关闭、外部搜索关闭、最多 80 次模型调用、600 秒、6 页视觉预算。结果见 `live-status.json` 和其指向的 `runs/<timestamp>/`；报告完整性检查见 `report-validation.json`。

本次未配置 embedding 模型，因此不构成真实向量入库测试。没有进行专家标注准确率评测，也没有执行 GPU 实验或复现论文速度数据。预算遗漏、核查未完成、模型不确定性必须以报告实际记录为准。

## 重跑结果（2026-11-21）

真实调用 `deepseek-flash` 49 次，49 次均有正常结束响应。记录 token 为输入 470,307、输出 128,515；仅计成功重跑，不含首轮失败和连通性探测，不能当作完整账单。

报告保存到 `runs/20261121T045212344750Z/report.md`，结构化数据为同目录 `research-report.json`。

37 条判断中：需人工复核 28、需限定条件 4、与当前证据不符 2、证据不足 3、verified 0。跨篇分析未执行。完成 2 个视觉转录片段，记录 127 条视觉观察（包含同页重复核验及不确定观察，不是 127 张图或 127 项通过）。候选页 4、7、8、9 未读取原图。

11 项引用/版本/范围等结构检查通过。`paragraphs` 为空，因此段落依赖检查为空集合检查，不能解释为 Writer 已经过真实测试；单篇 Markdown 来自判断卡渲染。

### 尚未解决、由本次真实 case 发现的问题

- 两批 Reader 输出 27、26 条判断，超出 25 条 schema 限制，被整批拒收。共有 39 个正文/OCR 片段被标记抽取未完成，不能说全文阅读完成；这不表示这些片段从未被作为核验上下文使用。
- 6 次定向修订返回多个判断，未满足单一判断要求，保留原判断及核查状态。
- 缺失页 8、9 的提示反复影响许多主张；需进一步区分目标真正依赖的图像与扩展检索带入的附带图像。6 页预算下选入大幅示例页而遗漏实验页，也值得调整优先级。
- 3 条证据不足的理由：视觉观察引用了未提供或版本不匹配的原图；视觉观察引用了未提供或版本不匹配的原图；1 validation error for VerificationDraft。完整校验错误以 JSON 为准。
- 模型将 LTX 的“stage-1 八步稠密、stage-2 使用 Sol-Attn”与“前八步稠密、其余三步使用 Sol-Attn”提示为潜在不一致，不能据此确认存在矛盾。两种叙述可能兼容，冲突标签仍需要人工复核。
- 模型正确保留了约 2 倍 Sol-Attn 加速与约 5 倍 Sol-Engine 组合优化的条件区别，但对 5.08、5.1、5 的舍入差别也表现出过度质疑倾向。

结论：真实模型与 PDF、视觉输入、证据引用、核验和报告持久化链路已运行；目前产物是有明确遗漏的初步分析，不是完整通过的论文报告。此次仅修复超时兜底和客户端关闭两个确定性程序问题，其余质量问题如实记录，未通过放松校验制造通过率。
