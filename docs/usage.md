# fin-rag 使用文档

本文档只描述当前仓库已经落地、可直接执行的能力。

## 1. 功能范围

当前可用功能：

- 扫描 `fin-online` 和 `opdata` 的 `docs/`、`.cursor/` 目录
- 使用本地 `BAAI/bge-base-zh-v1.5` Embedding 服务建索引
- 将向量库写入本地 Qdrant
- 通过 `query` 命令做融合检索，返回命中的文档块和元数据
- 通过 `ask` 命令走 CitationQueryEngine 做带引用问答

当前未接入：

- Workflow 三道闸
- Web API 和前端页面

## 2. 环境准备

### 2.1 依赖

项目要求 Python 3.11+。

推荐先创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev]
```

### 2.2 本地服务

需要先启动两个本地服务：

- Qdrant，默认地址 `http://localhost:6333`
- Embedding 服务，默认地址 `http://127.0.0.1:8001/v1`

Qdrant 可以通过 Docker 启动，Embedding 服务通过本仓库 CLI 启动。

## 3. 配置

默认配置文件：

- [config/settings.yaml](/Users/wangfei/yulore/fin-rag/config/settings.yaml)
- [config/sources.yaml](/Users/wangfei/yulore/fin-rag/config/sources.yaml)

常用配置项：

- `qdrant_url`
- `qdrant_collection`
- `embedding_base_url`
- `embedding_model`
- `embedding_dim`
- `embedding_batch_size`
- `llm_base_url`
- `llm_model`
- `llm_api_key`
- `llm_timeout`
- `llm_max_tokens`
- `llm_temperature`
- `docstore_path`
- `chunk_sizes`
- `similarity_cutoff`
- `fusion_top_k`

环境变量优先级高于 `settings.yaml`。当前代码支持的核心覆盖项是这些配置对应的 `FIN_RAG_*` 变量。

DeepSeek 推荐配置：

- `llm_base_url: https://api.deepseek.com`
- `llm_model: deepseek-v4-pro`
- `FIN_RAG_LLM_API_KEY` 仅放在本地环境变量里，不要写进仓库

## 4. 启动本地 Embedding 服务

Embedding 服务默认监听 `127.0.0.1:8001`，暴露 OpenAI-compatible 接口：

- `GET /v1/models`
- `POST /v1/embeddings`

启动命令：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli serve-embeddings
```

如果需要覆盖配置：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli serve-embeddings --settings path/to/settings.yaml
```

接口说明：

- `GET /v1/models` 返回当前模型列表
- `POST /v1/embeddings` 接收 `input`，支持字符串或字符串数组

## 5. 启动 Qdrant

默认 Qdrant 地址是 `http://localhost:6333`。

可用 Docker 启动后，再检查健康状态：

```bash
curl http://localhost:6333/healthz
```

如果返回 `healthz check passed`，说明服务可用。

## 6. 建索引

`index` 命令会：

- 扫描数据源
- 解析文档
- 层级切分
- 调用本地 Embedding 服务生成向量
- 写入 Qdrant
- 持久化 DocStore

执行命令：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli index
```

只做扫描统计，不写库：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli index --dry-run
```

自定义配置文件：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli index --settings path/to/settings.yaml
```

自定义数据源清单：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli index --sources path/to/sources.yaml
```

索引完成后，命令会输出类似信息：

```json
{
  "document_count": 233,
  "node_count": 6271,
  "collection_name": "fin_rag_docs",
  "qdrant_url": "http://localhost:6333"
}
```

## 7. 查询

`query` 命令当前返回的是检索结果，不是最终 LLM 答案。

执行命令：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli query "账期切换逻辑是什么"
```

可选 domain 过滤：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli query "账期切换逻辑是什么" --domain fin-online
```

`--domain` 可以重复传入：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli query "账期切换逻辑是什么" --domain fin-online --domain opdata
```

输出格式为 JSON 数组，每一项包含：

- `score`
- `text`
- `metadata`

## 8. 问答

`ask` 命令会走 CitationQueryEngine，并在返回前做引用/产品号校验；校验失败时会拒答。
当前仓库已经接好入口，但需要你在 `settings.yaml` 或环境变量里配置一个 OpenAI-compatible LLM 服务。

执行命令：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli ask "账期切换逻辑是什么"
```

可选 domain 过滤：

```bash
PYTHONPATH=src .venv/bin/python -m fin_rag.cli ask "账期切换逻辑是什么" --domain fin-online
```

如果没有配置 `llm_base_url`，命令会直接报错退出。

## 9. 数据源目录

当前默认索引两个仓库：

- `fin-online`
- `opdata`

对应路径在 `config/sources.yaml` 中配置。默认只索引：

- `docs/`
- `.cursor/`

已排除内容：

- `*bxf*`
- `docs/testcase/**`
- `docs/review/**`
- `.DS_Store`
- `target/**`
- `.git/**`

## 10. 常见问题

### 10.1 `query` 报错找不到 collection

先重新执行 `index`。

### 10.2 `index` 能跑完，但 `node_count` 为 0

通常表示没有生成 embedding 或没有把节点写进 Qdrant。先确认：

- 本地 Embedding 服务已启动
- `embedding_base_url` 配置正确
- Qdrant 服务可访问

### 10.3 `query` 没有返回结果

先检查：

- 传入的问题是否落在当前索引范围内
- `--domain` 是否过滤过窄
- `index` 是否刚刚成功写入

## 11. 建议的运行顺序

```bash
1. 启动 Qdrant
2. 启动 embedding 服务
3. 配置 LLM 服务
4. 执行 index
5. 执行 query 或 ask
```
