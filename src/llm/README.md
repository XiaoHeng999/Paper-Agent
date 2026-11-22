# 模型适配层

上层通过 `src.llm` 装配模型服务。聊天使用 Agent 配置，向量化使用独立的 embedding profile，两种配置都引用具名 Provider。配置字段见[项目示例](../../config/model.example.json)。

| 模块 | 职责 |
| --- | --- |
| [config.py](config.py) | 解析模型档位、embedding profile、Provider 地址和环境变量凭据 |
| [registry.py](registry.py) | 定义支持的协议和 Provider 的默认行为 |
| [factory.py](factory.py) | 根据配置创建 Provider，返回包含实例、模型名和配置签名的 `ProviderSnapshot` |
| [base.py](base.py) | 定义 Provider 接口、响应和流式回调 |
| [openai_compat.py](openai_compat.py) 和 [anthropic.py](anthropic.py) | 在统一消息结构与具体服务的请求和响应格式之间转换 |

修改适配器后，在仓库根目录执行 `uv run --locked python -m unittest test.test_llm_adapters -v`。需要验证真实服务连接时，参见[测试说明](../../test/README.md)。
