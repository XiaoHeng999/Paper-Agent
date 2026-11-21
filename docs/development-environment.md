# uv 开发环境

项目已有 `pyproject.toml`、`uv.lock` 和指定 Python 3.12 的 `.python-version`。在仓库根目录执行：

```bash
uv sync --locked
```

该命令根据现有锁文件创建或同步 `.venv`。日常通过 `uv run --locked` 执行 Python 命令，无需手动激活环境或重新运行 `uv init`。`.venv/` 已被 Git 忽略。

启动后端：

```bash
uv run --locked python main.py
```

真实论文任务还需要在系统设置中配置可用模型；依赖安装成功不代表模型 API 已验证。配置示例为 `config/model.example.json`，本地配置 `config/model.json` 被 Git 忽略。前端依赖单独使用根目录的 `npm run front:install` 和 `npm run front:dev` 管理。

本次环境配置结果：Python 3.12.15，104 个依赖包已安装；`uv pip check` 通过，FastAPI、LangGraph、ChromaDB、OpenAI、Anthropic 、pypdf、PDFium 与 Pillow 可用。以下离线适配器测试全部通过（4 项），使用模型替身，不调用真实 API：

```bash
uv pip check
uv run --locked python -m unittest test.test_llm_adapters -v
```

该检查只验证依赖和适配器基本行为，不代表整个论文工作流已端到端通过。开发新架构时按 [架构迁移计划](./multi-agent-architecture.md) 增加对应的对象、恢复和 handoff 测试；全部现有测试的运行方式见 `test/README.md`。

本次架构实施后，全量离线测试运行 149 项，其中 145 项通过，4 项真实模型测试按环境开关跳过；前端类型检查及 Vite 构建通过。当前系统自带 Node 12 不满足前端工具要求，本次使用临时 Node 22 执行前端验证。在 `front/` 内可复现：

```bash
npm exec --yes --package=node@22 -- node node_modules/vue-tsc/bin/vue-tsc.js --noEmit
npm exec --yes --package=node@22 -- node node_modules/vite/bin/vite.js build
```

日常前端开发建议使用 Node 22；真实模型测试分别通过 `RUN_LIVE_MODEL_TEST=1` 和 `RUN_MODEL_CONNECTIVITY_TEST=1` 显式开启，并要求有效的本地模型配置。

向量检索测试使用临时真实 Chroma 数据库和 embedding 替身，不访问真实模型服务；覆盖持久化复用、版本隔离、单篇范围、跨篇开关、全文自动接入和故障降级。

视觉、评测和增量功能的使用及验证边界见 [说明](./visual-evaluation-incremental.md)。真实模型评测已尝试启动，但本地未加载到可用研究模型，因此没有生成真实模型成绩；真实专家标注仍待人工提供。

最近一次深度审查修复新增 26 项回归测试（`test/test_review_fixes.py`），覆盖文档增页、陈旧来源、混合扫描、逐页视觉观察、关系进入写作与增量失效、适用维度评分、轮转候选调度和评测回避基线。
