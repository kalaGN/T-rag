# t-rag 使用说明

## 1. 安装与配置

Python 3.11+：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp config/settings.example.yaml config/settings.yaml
```

设置 `llm_base_url`、`llm_model`，通过 `T_RAG_LLM_API_KEY` 提供密钥。所有 Settings 字段中支持的环境变量均以 `T_RAG_` 开头，优先于 YAML。旧 `FIN_RAG_*` 变量不再读取。

默认数据目录 `data/t_rag`，每个知识库包含 JSON 清单、上传副本、DocStore；Qdrant 集合名称为 `t_rag_<知识库ID>`。旧金融集合和 `data/docstore` 不被自动迁移或删除。

模型和切分配置只通过 YAML/环境变量编辑，不提供页面设置。Embedding 模型、地址、维度或切分改变后必须重建。没有配置文件时模块 CLI 使用默认设置；启动脚本回退至示例配置。

## 2. 页面启动

```bash
scripts/dev.sh web
```

开发脚本检查 Qdrant（默认 6333），需要时启动 Docker 容器，并后台启动本地 BGE Embedding（默认 8001），前台运行 Gradio（默认 7860）。Qdrant 容器退出脚本后仍保留；脚本自己启动的 Embedding 随脚本退出。

只启动单个组件：

```bash
scripts/start.sh embeddings
scripts/start.sh web --host 127.0.0.1 --port 7860
```

可通过 `PYTHON_BIN`、`SETTINGS_PATH` 设置 Python 和配置路径。默认只监听本机，本版没有账号认证。

页面流程：

1. 在“知识库”创建名称，选择当前知识库。
2. 在“文档与重建”上传文件或登记目录，配置包含子目录与排除规则。包含 `.` 表示整个目录。
3. 点击“全量重建索引”，查看进度与结果。
4. 在“问答”输入问题，查看答案和引用；展开来源片段并预览已登记文档。
5. 更新同名上传文件、修改目录文件或移除数据源后，再手动重建。
6. 删除知识库需勾选确认；清理应用数据和对应集合，目录原始文件保留。

只有导入成功不会建索引。空知识库重建后可用但没有片段，问答会拒答。重建失败时请检查文档编码、Qdrant、Embedding，然后再次重建。

## 3. 支持格式

Markdown/MDC 保留 ATX 标题层级，MDC 去除 frontmatter；TXT 使用全文；可提取文字的 PDF 保留每页位置；CSV 和 XLSX 保留工作表/行范围。读取全部页面和行，不静默取前10页或10行。

输入文本使用 UTF-8；扫描 PDF 不支持 OCR，无法提取文字时报错。CSV/XLSX 按30行组织内容段，再按切分配置切块。只支持 `.md/.mdc/.txt/.pdf/.csv/.xlsx`。目录外符号链接不纳入数据源。

## 4. CLI

安装后可使用 `.venv/bin/t-rag`，或 `PYTHONPATH=src .venv/bin/python -m t_rag.cli`。

```bash
.venv/bin/t-rag kb create "产品文档" --settings config/settings.yaml
.venv/bin/t-rag kb list --settings config/settings.yaml
.venv/bin/t-rag source add --kb ID --directory /absolute/docs --include . --exclude '.git/**'
.venv/bin/t-rag source upload --kb ID --file /absolute/manual.pdf
.venv/bin/t-rag source list --kb ID
.venv/bin/t-rag source remove --kb ID --id SOURCE_ID
scripts/start.sh index --kb ID
scripts/start.sh query --kb ID "文档说了什么" --strategy fusion
scripts/start.sh ask --kb ID "文档说了什么"
.venv/bin/t-rag kb delete ID --confirm
```

`ID` 与 `SOURCE_ID` 替换为实际返回标识。`index/query/ask` 必须指定 `--kb`，支持 `vector/bm25/fusion` 三种策略。索引/问答可用同一 `--settings` 指定配置。旧 `--domain`、`--sources`、`serve-api` 已移除。

先停止页面再使用同一数据目录的 CLI；进程锁会拒绝第二个进程。测试可以使用不同 `T_RAG_DATA_DIR`。

## 5. 索引与回答边界

每个知识库一个集合，全量覆盖重建。重建前标记不可用，完成向量、DocStore 和来源校验后才标记可用；失败不保留旧版在线服务。应用重启将中断的重建标记失败。

来源和文件哈希、Embedding/切分签名在查询前校验，变化时要求重建。重复重建不会累积片段，移除来源后重建不再检索其内容。

向量检索用原始相似度门槛，BM25 用中文字符/双字及英文词项，融合采用 RRF 排序。编号引用必须存在且有效；无来源不调用 LLM，无引用或越界引用拒答。引用校验不证明答案事实全部正确，需核对原文。

不支持增量同步、版本发布、后台任务恢复、OCR、网页抓取、数据库查询、多轮会话或流式生成。
