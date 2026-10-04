# t-rag

本机单用户文档 RAG 系统，通过 Gradio 页面管理多个知识库并进行带引用问答，不提供独立业务 API。

## 启动

Python 3.11+，需要本地 Qdrant 和 Embedding 服务。

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp config/settings.example.yaml config/settings.yaml
scripts/dev.sh web
```

在本地配置文件填写 LLM 地址和模型，密钥通过 `T_RAG_LLM_API_KEY` 环境变量提供。页面创建知识库、导入文件或目录、全量重建后即可问答。远程 LLM 会收到检索文本。

文档或 Embedding 配置变更需重建；重建中及失败时不可查询。仅支持一个应用进程，CLI 与页面不能同时使用同一数据目录。

## 文档与验证

- [使用说明](docs/usage.md)
- [需求规格](docs/spec/2026-10-04-general-rag-spec.md)
- [实施计划](docs/plan/2026-10-04-general-rag-plan.md)
- [开发规则入口](AGENTS.md)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
T_RAG_INTEGRATION=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/integration -q
```

源码在 `src/t_rag`，按 knowledge、ingestion、retrieval、qa、models、storage、web 划分；单元测试按模块放入 `tests/unit`，真实服务测试放入 `tests/integration`。运行数据不纳入 Git。
