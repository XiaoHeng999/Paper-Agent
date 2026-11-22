# Paper-Agent

Paper-Agent 是面向科研学生和研究者的论文阅读工作台。给定研究主题，系统可以从 arXiv、OpenAlex 和 Semantic Scholar 检索相关论文。已有目标论文时，也可以通过 API 或 Python 指定论文及其 PDF 链接，直接进入阅读流程。

默认按篇生成阅读报告。报告先讲解研究问题、方法和算法流程，再结合原文分析关键假设、实验支持和结论的适用范围。读者可以沿报告里的证据引用回查原文。允许跨篇分析且至少有两篇可用论文时，可以进一步比较不同工作的主张、实验条件和分歧，或生成综述。为某个证据缺口额外搜索论文由单独的选项控制。

## 本地运行

准备 Python 3.12、uv 和 Node.js 22，在仓库根目录执行以下命令。前端构建完成后由 FastAPI 一起提供服务。

```sh
uv sync --locked
npm ci --prefix front
npm run front:build
uv run --locked python main.py
```

打开 [本地工作台](http://127.0.0.1:8000/)，在「系统设置」中填写模型服务地址、模型名称和凭据，测试连接后保存，再进入会话页提交研究主题。会话页可查看任务进度、历史记录和报告。

也可以参照 [模型配置示例](config/model.example.json) 创建本地 `config/model.json`。在 `agents.default_agent` 指定模型和对应的 Provider。如果密钥放在项目根目录的 `.env` 中，在 Provider 的 `api_key_env` 填写变量名，然后使用以下启动命令。向量检索可另行配置 embedding，未配置时仍可按原文片段阅读与核查。

```sh
uv run --locked --env-file .env python main.py
```

## 使用入口

| 场景 | 入口 |
| --- | --- |
| 交互式阅读 | 本地工作台提交主题并查看报告 |
| 接入已有应用 | [本地 API 文档](http://127.0.0.1:8000/docs) 提供会话、任务和设置接口 |
| 在脚本中分析指定论文 | 构造 `ResearchRequest`，在 `constraints.papers` 中传入含 `id`、`title`、`pdf_url` 的条目，然后调用 [run_graph_sync](src/graph/graph.py) |

## 开发与测试

后端使用 FastAPI 和 LangGraph，前端使用 Vue 3 和 TypeScript。开发前端时，保持后端运行，在另一个终端执行 `npm run front:dev`，打开 [前端开发页](http://127.0.0.1:5173/)。提交改动前可运行后端测试和前端类型检查及构建。

```sh
uv run --locked python -m unittest discover -s test
npm run front:build
```

本项目参考并沿用了 [Tswoen/Paper-Agent](https://github.com/Tswoen/Paper-Agent) 的代码。
