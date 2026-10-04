# 通用 RAG 首版实施计划

日期：2026-10-04。状态：收缩版首期已实施；验证结果见 [验证记录](../verification/general-rag.md)。依据：[收缩后的规格](../spec/2026-10-04-general-rag-spec.md)。本文件替代原12项完整产品计划。

## 技术决策

- 每库一份 JSON 清单、一个 Qdrant 集合和一份 DocStore，不引入 SQLite。
- 全量重建，重建期间暂停该库查询，失败显示不可用并允许重试，不保留在线旧版本。
- Gradio 队列执行长操作并显示进度，状态保存清单，不建立独立任务系统。
- 一把进程内锁协调写操作与索引查询；仅支持一个应用进程。重启识别中断，状态改为失败。
- 数据源删除和文档变化要求手动重建；查询前比较文件清单和哈希、索引配置签名。
- 通用核心保留原模型栈；模型设置只通过 YAML/环境变量配置。
- 原金融数据不迁移，任何覆盖或删除只能针对已登记的新系统集合与管理目录。

## 任务与依赖

### T1：知识库清单和管理
- [x] 创建、列表、选择、删除；JSON 原子保存，校验 ID 和管理路径。
- 文件：knowledge/service.py、knowledge/__init__.py、storage/catalog.py、tests/unit/knowledge/test_service.py。
- 验收/验证：持久化、重名、非法 ID、保存失败；删除失败可重试，原文件保持。

### T2：数据源导入
- [x] 上传复制、同名替换、目录包含/排除、来源移除、文件哈希清单。
- 文件：knowledge/service.py、ingestion/scanner.py、tests/unit/knowledge/test_sources.py。
- 依赖 T1。验收/验证：任意目录、bxf 默认可用、重复导入、不修改原文件；来源变更标记需要重建。

### T3：完整解析和定位
- [x] 分段读取六类格式，保留章节、页码和行位置，移除金融元数据。
- 文件：ingestion/readers.py、ingestion/documents.py、ingestion/metadata.py、tests/unit/ingestion/test_readers.py、tests/unit/ingestion/test_documents.py。
- 依赖 T2。验收/验证：真实多页PDF、多行XLSX/CSV、Markdown层级；不截断，错误不伪装成功。

### T4：手动全量重建
- [x] 状态控制、集合重建、叶节点入库、DocStore、配置签名、写查询锁及中断状态。
- 文件：knowledge/indexing.py、ingestion/llama_pipeline.py、storage/runtime.py、tests/unit/knowledge/test_indexing.py。
- 依赖 T3。验收/验证：空库、Embedding/写库/持久化失败、重试、中断；失败禁止查询部分索引。

检查点 A：管理和索引单元测试通过，本地集成重建成功。

### T5：统一检索
- [x] 显式知识库选择、三种策略、中文词项匹配、双路隔离、原始分数门控。
- 文件：retrieval/fusion.py、retrieval/filters.py、retrieval/query_rewrite.py、tests/unit/retrieval/test_retrieval.py、tests/integration/test_index_lifecycle.py。
- 依赖 T4。验收/验证：两库同名同词隔离、重建后旧片段消失、哈希/签名变化拒绝查询。

### T6：引用问答
- [x] 统一检索后生成，移除产品号规则及取消门槛重试，提供通用来源。
- 文件：qa/workflow.py、qa/citation.py、qa/prompts.py、retrieval/query_engine.py、tests/unit/qa/test_workflow.py。
- 依赖 T5。验收/验证：有效/越界/无引用、空候选不调用模型、普通产品数字不误拒答。

### T7：配置和 CLI
- [x] T_RAG_* 配置、data_dir、t-rag 命令、知识库管理与显式 --kb，脚本统一配置透传。
- 文件：config.py、cli.py、scripts/start.sh、pyproject.toml、tests/unit/config/test_config.py。
- 依赖 T6。验收/验证：临时库创建/列表、query/ask 参数、秘密不回显；验证可安装入口。

### T8：页面管理
- [x] 知识库选择/创建/删除确认、来源上传/目录/移除、重建进度及状态。
- 文件：web/app.py、web/management.py、tests/unit/web/test_management.py。
- 依赖 T7。验收/验证：真实浏览器操作，失败可见且可重试，无模型设置区域。

### T9：问答页面
- [x] 问答与引用预览、展开片段；预览只读取登记来源。
- 文件：web/app.py、web/qa.py、tests/unit/web/test_web.py、tests/e2e/test_web_flow.py。
- 依赖 T8。验收/验证：浏览器完整闭环、空库/需重建提示、任意路径不可读。

### T10：文档与交付
- [x] 更新示例配置、使用文档、README、验收记录。
- 文件：README.md、docs/usage.md、config/settings.example.yaml、config/sources.yaml、docs/verification/general-rag.md。
- 依赖 T9。验证：全量测试、脚本/CLI、双库实际集成与浏览器，记录真实 LLM 是否调用，不以 Mock 宣称端到端通过。

## 验证和限制

全库重建和查询前哈希检查适合本地小到中规模文档，记录耗时后再考虑优化。单用户仍可能并发提交请求，必须在服务层协调锁，不能只靠按钮禁用。配置签名不包含密钥。

既有虚拟环境缺 hatchling，实施时处理构建验证；未验证的本地服务、真实 LLM 或浏览器必须明确记录。现有金融索引保持独立，不自动导入。


实施记录：长操作使用 Gradio 队列；知识库后台任务、SQLite 和版本发布未引入。BM25 以本地中文分词及标准公式实现，避免默认英文分词。浏览器流程由 CUA 在本机手动验证，步骤和截图保存于验证文档，无新增浏览器测试依赖。移除数据源后上传副本保留到知识库删除，避免误删和重复恢复流程。
