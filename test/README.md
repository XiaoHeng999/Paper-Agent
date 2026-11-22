# 运行测试

在仓库根目录同步锁定的依赖，再运行 Python 标准库的 unittest 测试发现。

```sh
uv sync --locked
uv run --locked python -m unittest discover -s test -v
```

开发时可只运行相关模块。例如，下面的命令分别检查论文研究流程和模型协议适配器。

```sh
uv run --locked python -m unittest test.test_research_workflow -v
uv run --locked python -m unittest test.test_llm_adapters -v
```

真实模型测试默认跳过。要实际请求服务，先按[项目说明](../README.md)配置本地模型。以下为使用 Bash 和本地 `.env` 的示例。

```sh
RUN_MODEL_CONNECTIVITY_TEST=1 uv run --locked --env-file .env python -m unittest test.test_model_connectivity -v
RUN_LIVE_MODEL_TEST=1 uv run --locked --env-file .env python -m unittest test.test_live_model_config -v
```
