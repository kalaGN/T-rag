# fin-rag 设计方案：运营/金融知识库智能问答助手

- **状态**：实施中（索引、检索与最小 QA 编排已落地）
- **日期**：2026-06-24
- **作者**：王飞 + Codex
- **项目名**：`fin-rag`
- **项目根目录**：`/Users/wangfei/yulore/fin-rag`

---

## 1. 概述

### 1.1 目标

基于 `fin-online.dianhua.cn` 与 `opdata.dianhua.cn` 两个项目的 `docs/` 与 `.cursor/` 目录构建一个**团队共享的 Web 智能问答助手**，让研发/测试/产品能自然语言查询业务逻辑、技术方案、渠道对接、工程规则等知识，并能**精确溯源**到原始文档（文件 + 章节），在金融业务场景下**对幻觉零容忍**。

### 1.2 非目标（首版不做）

- 不接入 MySQL（预留 `DataSource` 接口，二期再做）
- 不做业务域权限隔离（`domain` 仅作检索筛选项，全库默认可见）
- 不纳入 `shield / jindun / hmf / sjf / bxf` 的独立业务域治理（打标规则预留识别，二期治理）
- 不替换现有任何系统，仅为只读问答

### 1.3 关键约束

| 约束 | 取值 |
|------|------|
| 形态 | 团队共享 Web 服务（首期 Gradio，二期 Vue） |
| LLM 与隐私 | Embedding 使用本地 `BAAI/bge-base-zh-v1.5` 服务，文档全文不发送到第三方向量化厂商；LLM 生成阶段支持 DeepSeek/OpenAI-compatible 服务 |
| 技术栈 | Python 3.11+ |
| 检索范围 | 两项目统一索引（全量重建，幂等 `chunk_id`），后续可配置加入其他项目 |
| 数据源 | 本地文件系统（`docs/` + `.cursor/`） |

### 1.4 当前已落地范围

已完成：

- 本地 Embedding 服务：`BAAI/bge-base-zh-v1.5`
- Qdrant Docker 独立服务
- `fin-online` / `opdata` 文档扫描、解析、切分与索引
- DocStore 持久化与重载
- 融合检索链路：向量检索 + BM25
- `fin-rag index` 与 `fin-rag query` CLI
- `fin-rag ask` CitationQueryEngine 问答入口
- OpenAI-compatible LLM 包装器（可对接 DeepSeek 等兼容服务）

待接入：

- Web API / Gradio 前端
- 评测集与自动回归

---

## 2. 首版范围锁定

| 项 | 内容 |
|----|------|
| **数据源** | `fin-online.dianhua.cn` + `opdata.dianhua.cn`（均采集 `docs/` 与 `.cursor/`） |
| **采集排除规则** | 路径/文件名匹配 `*bxf*`（大小写不敏感）一律跳过不索引；`docs/testcase/` 与 `docs/review/` 目录完全排除（不索引测试用例与代码评审）；同时排除 `.DS_Store`、`target/`、`.git/`、二进制文件；超大 CSV（如 9MB 用例模板）仅建摘要、不入全文 |
| **业务域（domain）首版** | `fin-online`（通用）、`opdata`；打标规则同时识别 shield/jindun/hmf/sjf/bxf，供二期治理 |
| **不采集的文档类型** | 测试用例（`docs/testcase/`）、代码评审（`docs/review/`）—— 不入库、不索引 |
| **MySQL** | 预留可插拔 `DataSource` 接口（二期）；数据路由二期随 MySQL 一起加，首版无 |
| **隔离** | 不隔离，`domain` 仅作检索筛选项 |

> **关于 bxf**：bxf 知识全部位于 `fin-online` 仓库内部（`docs/plan/bxf/**`、`*bxf*.md`、`测试用例-bxf-*` 等）。经确认，首版**完全排除**所有含 `bxf` 的文件，不索引。

---

## 3. 数据源画像（已探查）

| 项目 | 规模 | 核心内容 |
|------|------|---------|
| `fin-online.dianhua.cn` | ~345 文件（主力） | `.cursor/rules` 11 个 `.mdc` 工程规则；`.cursor/memory/special` 业务逻辑（账期切换、降查得、日志三级备份、线上产品 PID 列表…，**最稀缺 know-how**）；`docs/spec` 需求、`docs/plan` 技术方案、`docs/testcase`、`docs/review`、`docs/channel-optimization`、`docs/apifox`、`docs/importcase`、`docs/diagrams` |
| `opdata.dianhua.cn` | ~10 文件 | `docs/channels` 渠道接口文档（LanChen 系列）+ 渠道配置 CSV；`.cursor/rules/anti-hallucination.mdc`；`memory/project.md` |

**数据特征（影响设计）**：

- **中文为主**，领域术语密集（账期、降查得、PID、备源、补数、渠道分流、产品号 3xxxxx/6xxxxx…）
- **混合格式**：`md` + `mdc`（带 YAML frontmatter 的 Cursor 规则）+ `csv/xlsx`（配置/用例）+ `pdf/java/xml`（样本）
- **文档强关联**：需求 → 技术方案 → 测试用例 → 代码评审，天然适合引用溯源与多跳检索
- `memory/special` 与 `.cursor/rules` 是**最该被精准召回的稀缺内容**

---

## 4. 系统架构总览

五层架构，核心设计原则：**LlamaIndex 主导、结构化优先、全链路溯源、多项目统一检索、防幻觉三道闸**。

```
┌─────────────────────────────────────────────────────────────────┐
│                         接入层 (Access)                          │
│   团队 Web 站点 (FastAPI + 首期 Gradio / 二期 Vue)              │
│   问答历史 · 引用溯源面板 · 点赞点踩反馈 · 业务域筛选             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      问答编排层 (QA Orchestration)               │
│  LlamaIndex Workflow: 检索 → 门控 → CitationQueryEngine         │
│  → 引用事后校验 → 返回                                          │
│  · 系统提示硬约束"只基于引用作答、无依据则明说"                 │
│  · 每条结论标注 [domain/文件/章节] 溯源                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     检索层 (Retrieval)                           │
│  QueryFusionRetriever: 稠密 + BM25 → RRF → AutoMergingRetriever │
│  → NodePostprocessor 业务加权/重排(metadata 过滤 domain)        │
│  · 神经重排(BGE-Reranker)首版不上，评测后按需加                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     索引层 (Indexing)                            │
│  VectorStoreIndex + QdrantVectorStore · DocStore · SQLite       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  数据管道 (Pipeline)  [LlamaIndex IngestionPipeline]             │
│  Reader(含排除规则) → Document → Transformations                │
│  → HierarchicalNodeParser → MetadataExtractor → Embedding       │
│  → QdrantVectorStore + DocStore（doc_id/hash 幂等）              │
│  数据源: DataSource 抽象(FilesystemDataSource 首版;             │
│           MySQLDataSource 二期)                                 │
│  注: 首版全量重建即可；增量同步二期(文档高频更新时再做)         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 关键组件清单（高内聚、单一职责、可独立测试）

| 组件 | 职责 | 关键依赖 |
|------|------|---------|
| `SourceReader` | 实现 LlamaIndex `BaseReader`，扫描多项目目录、应用排除规则并输出 `Document` | LlamaIndex Reader |
| `FormatReader` | 按 `.md/.mdc/.csv/.xlsx/.pdf` 解析为 LlamaIndex `Document`，保留章节和 frontmatter | markdown-it / openpyxl / pypdf |
| `IngestionPipeline` | 串联清洗、层级切分、元数据提取、向量化、缓存与写库 | LlamaIndex |
| `HierarchicalNodeParser` | 生成父/子 `TextNode`，小块检索、父块回灌 | LlamaIndex |
| `BusinessMetadataExtractor` | 自定义 `TransformComponent`，抽取 project/domain/doc_type/产品号/渠道/章节 | LlamaIndex + 正则规则 |
| `Embedding` | 通过 LlamaIndex 接入本地 `BAAI/bge-base-zh-v1.5` Embedding 服务（HTTP/OpenAI-compatible） | LlamaIndex |
| `VectorStoreIndex` | 统一索引入口；向量与 metadata 写入 `QdrantVectorStore`，父节点写入 DocStore | LlamaIndex + Qdrant |
| `QueryFusionRetriever` | 融合向量检索与 BM25 检索，采用 RRF | LlamaIndex |
| `AutoMergingRetriever` | 命中足够多子节点时回灌父节点，补全上下文 | LlamaIndex |
| `NodePostprocessor` | domain 过滤、业务权重、BGE 重排和最低相关性门控 | LlamaIndex + 自定义扩展 |
| `CitationQueryEngine` | 基于检索节点生成带编号引用的答案（待接入） | LlamaIndex |
| `QAWorkflow` | 编排检索、门控、生成、引用校验和错误处理（待接入） | LlamaIndex Workflow |
| `WebApp` | API + 前端，问答历史/反馈/业务域筛选 | FastAPI + Gradio |

> **首版不含**（二期随 MySQL 一起加）：`Router`/`DataQueryProvider`（文档/数据路由）和目录监听服务。首版由 `IngestionPipeline + IngestionCache + doc_id` 处理重复与变更；实时监听和删除事件同步留到文档高频更新时再上。

---

## 5. 数据管道与切分策略

### 5.1 统一中间结构：LlamaIndex `Document` / `TextNode`

不再自建 `Document/Node` 领域模型。Reader 产出 LlamaIndex `Document`，IngestionPipeline 产出 `TextNode`；业务字段统一放入 `metadata`，父子关系使用 LlamaIndex 原生 relationship。每个节点都带富元数据，作为过滤与溯源抓手：

```python
{
  "text": "...",                       # 切块正文
  "project": "fin-online",             # 物理来源(工程/目录)
  "domain": "fin-online",              # 业务域(核心过滤维度)
                                       #   首版枚举: fin-online | opdata
                                       #   识别预留: shield | jindun | hmf | sjf | bxf
  "sub_domain": null,                  # 可选二级(如盾分/通信指数)
  "parent_product": null,              # 父产品归属(二期,如 comix→bxf)
  "source_type": "cursor_memory",      # doc | cursor_rule | cursor_memory | channel_doc | code_sample
  "doc_type": "business_logic",        # spec需求 | tech_plan | arch | business_logic | engineering_rule | channel_doc
                                       #   注: testcase/review 不采集,无此类型
  "product_code": "300007",            # 从文件名/标题/正文抽取的产品号(可空)
  "channel": "cucc",                   # cucc/cmcc/ctcc/LanChen/rong360…(可空)
  "file_path": "docs/plan/...",        # 原文回链
  "heading_path": ".../2.接口设计/2.3鉴权",  # 章节面包屑,溯源精确到段落
  "file_hash": "...", "mtime": "...",  # 幂等 chunk_id 用
  "chunk_index": 3,
  "doc_id": "fin-online:docs/plan/xxx.md",
  "parent_node_id": "..."
}
```

### 5.2 格式 Reader —— 一格式一策略

| 格式 | 处理策略 |
|------|---------|
| `.md` | markdown-it 解析，**保留标题树**作为切分骨架 |
| `.mdc`（Cursor 规则）⭐ | **剥离 YAML frontmatter**（`description/globs/alwaysApply`）入 metadata；正文按普通文档入库，用于检索与引用 |
| `.csv` | 配置/映射类 → 按行成组（表头做字段说明）；**大表（如 9MB 用例模板）→ 仅建摘要索引**，不入全文 |
| `.xlsx` | 按 sheet 成独立文档，大表抽摘要 + 关键列 |
| `.pdf` | 页/段切分 |
| `.java/.xml/.py`（混入样本） | 按 类/方法 切，标 `code_sample`，低权重 |
| `.DS_Store`/二进制/`target/`/`.git/` | 排除 |

### 5.3 NodeParser 切分策略——结构化优先

1. **标题树预切分**：Markdown Reader 保留 heading 层级，每段写入 `heading_path`。
2. **层级节点切分（Small-to-Big）**：使用 `HierarchicalNodeParser` 生成父子节点；叶子节点写入 VectorStore，完整节点写入 DocStore；查询时由 `AutoMergingRetriever` 回灌父节点。专门解决「账期切换逻辑」「日志三级备份」等长链路上下文断裂。
3. **块大小**：中文按 token，目标 300–500/块，重叠 50–80。
4. **稀缺内容加权**：`memory/special`（业务 know-how）切得更细，检索时 metadata boost，确保最稀缺的知识被优先召回。

### 5.4 元数据提取规则（多源推断）

- **project**：取 `sources.yaml` 配置的 `name`
- **domain**：按下表判定
- **doc_type**：按目录推断（`plan→tech_plan`、`spec→需求(spec)`、`memory/special→business_logic`、`rules→engineering_rule`、`channels→channel_doc`、`项目架构分析→arch`…）；**`testcase/review` 目录已排除采集，不会产生对应 doc_type**
- **产品号**：文件名（如 `300007`）+ 标题 + 正文「产品/PID 3xxxxx/6xxxxx」正则抽取
- **渠道**：文件名/标题关键词（cucc/cmcc/ctcc/LanChen/rong360/bxf…）

**domain 打标规则（核心过滤维度）**：

| 判定信号 | 规则 | 例子 |
|---------|------|------|
| 默认 | `sources.yaml` 的 `default_domain` | fin-online 仓默认 → fin-online |
| 目录 | `docs/plan/bxf/**`→bxf；`shield*`→shield；`opdata*`→opdata | bxf 迁移方案（注：首版 bxf 已排除采集，规则仅预留） |
| 文件名 | 含 `bxf/shield/jindun/hmf/sjf`（大小写不敏感） | `*jindun*.md`→jindun |
| 标题/正文 | 含上述关键词或对应产品号 | "金盾/号码风险等级"→jindun |
| 父子归属 | 二期：comix/comsc/610002/800-807 等 → `parent_product=bxf` | 迁移类文档 |
| 兜底 | 都不中 → `domain=default_domain`，并记入 `uncertain` 队列供人工确认 | — |

### 5.5 采集排除规则（首版硬性）

- 路径/文件名匹配 `*bxf*`（大小写不敏感）→ **跳过不索引**
- `docs/testcase/`、`docs/review/` 目录 → **完全排除不索引**（测试用例、代码评审不纳入问答库）
- `.DS_Store`、`target/`、`.git/`、二进制 → 排除
- 超大 CSV（阈值可配，默认 > 1MB）→ 仅摘要，不入全文

### 5.6 全量重建与幂等

- `fin-rag index` 调用 `IngestionPipeline.run(documents=...)`；持久化 `IngestionCache` 与 DocStore。
- `doc_id = {project}:{relative_path}`，文档 hash 用于判断重复、变更和删除；节点 id 由 `doc_id + transform_version + chunk_index` 稳定生成。
- 首版保留 `--rebuild` 全量重建；默认索引命令利用 ingestion cache 跳过未变化文档。无需另写一套自定义同步管道。

---

## 6. 检索层

### 6.1 流程

```
用户问题
   │
   ▼
[查询预处理] 查询改写 · 中文术语归一化("账期"↔"结算周期")
   │
   ▼
[融合检索]  ┌────────────────────┬──────────────────┐
            │ VectorIndexRetriever│ BM25Retriever   │
            └─────────┬──────────┴────────┬─────────┘
                      └ QueryFusionRetriever(RRF) ──┘
   │
   ▼
[父子回灌] AutoMergingRetriever：命中小块 → 回灌父章节
   │
   ▼
[后处理] SimilarityPostprocessor 门控 → 业务加权 → BGE 精排
   │
   ▼
[上下文组装] top-k 文档块 + 各自元数据 → 带 [domain/文件/章节] 溯源标签
```

### 6.2 关键设计点

1. **为什么必须融合检索**：语义类问题靠 `VectorIndexRetriever`；产品号、PID、渠道和表名等精确标识靠 `BM25Retriever`。由 `QueryFusionRetriever` 做 RRF，避免把 Qdrant 私有查询 API 扩散到业务层。BM25 中文分词效果必须通过 Golden Set 验证；若效果不达标，再切换 Qdrant hybrid/BM42，Retriever 接口不变。
2. **查询改写**：术语归一化表（账期↔结算周期↔billing cycle、降查得↔降档↔degrade…）——应对"用户口语 vs 文档术语"对不上。多查询展开二期再加。
3. **业务加权重排**：以自定义 `BaseNodePostprocessor` 实现 metadata 加权，再接 `SentenceTransformerRerank`。规则文档与其他文档一样参与检索，不再做默认硬注入。
4. **domain 过滤（仅筛选不隔离）**：默认全库搜索；前端提供业务域筛选器（fin-online / opdata / 全部）。选定域 → `filter domain in [...]`；未选 → 不带 filter。
5. **全程引用溯源**：上下文每块带溯源标签 `[domain / 文件路径 / §章节]`。

---

## 7. 问答编排与提示工程（防幻觉 + 溯源）

### 7.1 编排流程

> 当前仓库已完成索引与检索链路，`query` 命令目前输出检索结果和元数据；本节描述的是下一步要接入的问答编排目标态。

```
用户问题 + 历史
   │
   ▼
[检索取上下文] QueryFusionRetriever → AutoMergingRetriever → NodePostprocessors
   │
   ▼
[相关性门控] ①  若 top-k 分数全低于阈值 → 走"拒答模板"，不硬编
   │
   ▼
[提示组装] CitationQueryEngine + 自定义 PromptTemplate
   │
   ▼
[LLM 生成]  LlamaIndex DeepSeek/OpenAI-compatible LLM，强制带引用
   │
   ▼
[后处理校验] ②③ 引用校验(回贴来源真实) · 敏感字段检查 · 产品号/数字核对
   │
   ▼
[返回]  答案 + 引用列表(可点击回链) + 反馈(👍👎/纠错)
```

### 7.2 防幻觉三道闸（对齐 `anti-hallucination.mdc`）

| 闸 | 机制 | 实现 |
|----|------|------|
| **① 检索门控** | 召回置信度不足直接拒答 | `SimilarityPostprocessor` + Workflow 条件分支 |
| **② 生成约束** | 仅依据上下文作答，每条结论带引用 | `CitationQueryEngine` + 自定义 `PromptTemplate`（待接入） |
| **③ 事后校验** | 引用必须来自本次 `source_nodes`，产品号/数字与原文一致 | Workflow 自定义校验 Step；失败则拒答或重生成一次 |

### 7.3 系统提示骨架（防幻觉 + 溯源 + 中文金融口吻）

```text
你是 {domain} 业务知识助手。严格遵循：
1. 只基于<检索上下文>作答，不使用训练记忆，不臆测。
2. 每条结论后用 [n] 标注引用，对应文末来源；无依据的问题明确回答
   "知识库未覆盖"，绝不编造产品号/PID/渠道/账期/逻辑。
3. 涉及配置/SQL/接口时，给出文件路径与章节，必要时贴原文片段。
4. 业务术语用文档原词；金额/比例/产品号原样引用，不做四舍五入或推断。
5. 若多份文档冲突，列出各来源与差异，交由用户判断，不擅自裁决。

<检索上下文>
[1] [fin-online|plan/xxx技术方案.md|§2.3] …原文…
[2] [fin-online|.cursor/memory/special/账期切换逻辑.md|§1] …原文…
</检索上下文>
```

### 7.4 多轮与会话

- **历史压缩**：长对话滑窗 + 旧轮摘要，避免上下文爆炸
- **指代消解**：用 LLM 做一轮 query 改写（"它的鉴权"→"XXX 的鉴权"）再检索
- **会话隔离**：按用户/会话 id 隔离历史

### 7.5 失败模式与兜底

| 场景 | 兜底 |
|------|------|
| 召回不足（门控触发） | 拒答 + 给"相近问题/可补充文档"建议 |
| LLM 编造引用 | 事后校验拦截，标记 `低置信` 或重生成 |
| 超长答案 | 分段 + 结构化（结论先行 + 详情折叠） |
| LLM/Embedding API 失败 | LLM 调用重试；本地 Embedding 服务不可用时阻断索引/检索并告警，不回退到云端 Embedding |

---

## 8. 工程实现

### 8.1 技术栈定型

| 层 | 选型 | 理由 |
|----|------|------|
| 语言 | Python 3.11+ | 生态成熟，已选定 |
| RAG 框架 | **LlamaIndex** | Reader、IngestionPipeline、NodeParser、Index、Retriever、QueryEngine、Workflow 全链路统一 |
| 向量库 | **Qdrant**（首版即使用 Docker 独立服务） | 与团队共享部署形态一致，便于持久化、排障和后续迁移到内网服务器；原生支持 dense+sparse 混合检索 + RRF |
| Embedding | **`BAAI/bge-base-zh-v1.5` 本地服务**（OpenAI-compatible HTTP 接口） | 768 维，中文检索效果与资源占用平衡较好；适合当前 `Apple M1 Pro / 16GB` 机器先落地，文档不出本机/内网 |
| 重排 | **BGE-Reranker-v2-m3** | 中文精排强 |
| 稀疏检索 | **LlamaIndex BM25Retriever**（首版） | 与 `QueryFusionRetriever` 原生组合；效果不足可替换为 Qdrant hybrid/BM42 |
| LLM | **DeepSeek**（LlamaIndex DeepSeek 或 OpenAI-compatible 集成） | 仅答案生成走云端；Embedding 留在本地，降低文档外发范围 |
| 后端 | FastAPI + Pydantic | API + 输入校验 |
| 前端 | 首期 **Gradio**（`ChatInterface` 原生多轮对话 + 流式，溯源/反馈用自定义组件补），二期 Vue | RAG 聊天形态 Gradio 比 Streamlit 更省事（多轮/流式开箱即用） |
| 测试 | pytest，**目标 ≥80%** | 对应测试规范 |

### 8.2 项目结构（many small files / 高内聚低耦合）

```
fin-rag/
├── config/
│   ├── settings.yaml          # 模型/阈值/路径(无硬编码,全走配置)
│   └── sources.yaml           # 多项目数据源清单(可扩展加项目)
├── src/fin_rag/
│   ├── config.py              # Pydantic 配置加载+校验
│   ├── llama_settings.py      # LlamaIndex Settings: LLM/Embedding/Callback
│   ├── ingestion/             # ── LlamaIndex 数据管道
│   │   ├── readers/           # BaseReader: filesystem/markdown/mdc/tabular/pdf
│   │   ├── metadata.py        # TransformComponent: domain/doc_type/产品号
│   │   ├── pipeline.py        # IngestionPipeline + cache + docstore
│   │   └── index.py           # StorageContext + VectorStoreIndex
│   ├── retrieval/             # ── 检索
│   │   ├── query_rewrite.py   # 术语归一化
│   │   ├── fusion.py          # QueryFusionRetriever + AutoMergingRetriever
│   │   └── postprocessors.py  # 门控 + 业务加权 + BGE 重排
│   ├── qa/                    # ── 问答编排
│   │   ├── prompts.py         # 系统提示(防幻觉硬约束)
│   │   ├── citation.py        # 引用上下文与引用校验
│   │   └── workflow.py        # LlamaIndex Workflow 编排三道闸（待接入）
│   ├── store/                 # ── 存储
│   │   ├── qdrant.py          # QdrantVectorStore/StorageContext 工厂
│   │   └── metadata_store.py  # SQLite: 反馈/问答历史
│   ├── api/                   # server.py / routes.py / schemas.py
│   └── web/app.py             # Gradio 首期前端 (ChatInterface)
├── eval/                      # golden_set.jsonl + metrics + run_eval
├── tests/                     # pytest ≥80%
├── docker-compose.yml         # app + qdrant
└── pyproject.toml
```

**编码风格约束**（对应团队 coding-style）：

- **不重复造模型**：索引链路使用 LlamaIndex `Document/TextNode/NodeWithScore/Response`；API DTO 才使用 Pydantic
- **每文件 200–400 行，800 上限**，按 domain 组织
- **全链路 try/except**，抛出用户友好错误信息
- **Pydantic 校验所有入参**
- **无 `print` 残留**，统一 logging；**无硬编码**，全走 `config/`

### 8.3 配置驱动（加项目零改代码）

```yaml
# config/sources.yaml
sources:
  - name: fin-online
    root: /Users/wangfei/yulore/fin-online.dianhua.cn
    include: [docs/, .cursor/]
    exclude_patterns: ["*bxf*", "docs/testcase/", "docs/review/", ".DS_Store", "target/", ".git/"]
    default_domain: fin-online
  - name: opdata
    root: /Users/wangfei/yulore/opdata.dianhua.cn
    include: [docs/, .cursor/]
    default_domain: opdata
# 二期加项目：在此追加一项即可
```

### 8.4 评测

- **Golden Set**：从现有文档手工构造 30–50 条金标问答（覆盖账期切换/降查得/日志三级备份/渠道配置/anti-hallucination 规则等），标注期望要点 + 期望引用文件
- **指标**：使用 LlamaIndex retrieval/response evaluators 计算召回、忠实度和相关性；自定义计算引用准确率与拒答准确率
- **回归**：每次改切分/检索参数自动跑评测

关键验收场景：

| 场景 | 触发条件 | 预期渠道状态 | 预期产品状态 | 日志验证点 |
|------|----------|--------------|--------------|------------|
| 精确编号检索 | 查询具体 PID/产品号 | 不涉及渠道 | 返回包含正确编号的文档节点 | fusion 两路结果、最终 source_nodes |
| 长业务逻辑 | 查询账期切换/日志备份 | 不涉及渠道 | 返回完整父章节，不只返回碎片 | AutoMerging 前后节点 id |
| domain 筛选 | 指定 opdata | 不涉及渠道 | source_nodes 全部属于 opdata | metadata filter 条件 |
| 无依据问题 | top-k 低于阈值 | 不进入渠道 | 明确拒答，不调用或不采纳生成结果 | 门控分数、拒答原因 |
| 引用校验失败 | 答案引用不存在 | 不涉及渠道 | 重生成一次，仍失败则拒答 | source_nodes 与引用编号差异 |

### 8.5 配置变更清单

- `pyproject.toml`：增加 LlamaIndex core、Qdrant、BM25、reranker、DeepSeek/OpenAI-compatible、local embedding client 等按需拆分包；版本必须锁定。
- `config/settings.yaml`：增加 `chunk_sizes`、`similarity_cutoff`、`fusion_top_k`、`rerank_top_n`、`ingestion_cache_path`、`docstore_path`、`llm_base_url`、`llm_model`、`llm_api_key`、`llm_timeout`、`llm_max_tokens`、`llm_temperature`。
- 环境变量：`DEEPSEEK_API_KEY`、`QDRANT_URL`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`（如服务鉴权则再加 `EMBEDDING_API_KEY`），禁止写入配置文件。

建议默认值：

- `EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5`
- `EMBEDDING_DIM=768`
- `EMBEDDING_BATCH_SIZE=8`（本机 `M1 Pro 16GB` 起步值；离线全量索引可视负载逐步调到 16）

### 8.6 代码变更清单

- 删除自建 `Document/Node`、手写 Qdrant RRF 查询和自建 `QAChain` 的设计。
- 新增 LlamaIndex Reader、TransformComponent、IngestionPipeline、StorageContext、Retriever、NodePostprocessor、CitationQueryEngine、OpenAI-compatible LLM 工厂和 Workflow 工厂。
- 业务自定义代码仅保留：采集排除、格式清洗、metadata 规则、术语归一化、门控策略、业务加权、引用真实性校验。

### 8.7 风险与兼容性

- LlamaIndex 包拆分多、API 迭代快：锁定依赖版本，升级必须跑 Golden Set。
- 本地 Embedding 服务需要稳定暴露统一维度和模型版本；模型切换或维度变化必须触发全量重建索引。
- 本地 Embedding 吞吐受机器资源影响明显；批量索引时需限制并发和 batch size，避免把 CPU/GPU 打满影响在线问答。
- `BAAI/bge-base-zh-v1.5` 适合当前单机开发与小规模团队试用；若后续文档规模明显扩大或并发升高，再升级到更强模型或把 Embedding 服务拆到独立机器。
- `BM25Retriever` 需要可加载全部叶子节点；当前数百文档可接受，规模扩大后切 Qdrant hybrid/BM42。
- `AutoMergingRetriever` 依赖 DocStore 中存在父节点；索引与检索必须共享同一 StorageContext。
- 现有 M0-M5 实施计划基于自建组件，与本方案不兼容，已标记废弃，编码前需按本方案重新拆分。

### 8.8 部署

首版统一采用 Docker Compose 部署 `qdrant` + `app`，开发、测试、内网部署保持同构；默认通过 `QDRANT_URL=http://localhost:6333` 连接独立 Qdrant 服务。配置与索引数据卷持久化；API key 走环境变量（**绝不入库/硬编码**）。

### 8.9 里程碑

| 阶段 | 内容 | 周期 |
|------|------|------|
| M0 | 脚手架 + 配置 + 数据源清单 | 1d |
| M1 | LlamaIndex Reader + IngestionPipeline + 层级切分 + metadata + VectorStoreIndex | 3–4d |
| M2 | QueryFusionRetriever + AutoMergingRetriever + postprocessors | 2–3d |
| M3 | Workflow + CitationQueryEngine + 三道闸 + Gradio | 2–3d |
| M4 | 评测 golden set + 回归 | 1–2d |
| M5 | Golden Set 评测 + Docker 部署 + 团队验收 | 1–2d |

---

## 9. 二期规划

- **MySQL 数据源 + 文档/数据路由**：随 MySQL 加 `MySQLDataSource` 与 `Router`/`DataQueryProvider`（数据类问题改走 NL2SQL，只读账号 + 表/视图白名单 + 强制 LIMIT + 敏感字段脱敏 + 执行超时）；先把配置库表结构导成 schema 文档走 schema-RAG（最轻）
- **增量同步增强**：首版使用 IngestionCache/doc_id 去重；文档高频更新后再补目录监听、删除事件处理与定时任务
- **散落业务域治理**：把 `hmf / jindun / sjf / shield` 在 `testtool / apidoc / hmr` 等代码目录中的散落知识纳入采集，补全打标
- **bxf 独立域**：取消 `*bxf*` 排除，作为独立业务域纳入（含父产品归属）
- **Vue 正式前端** + **可选权限隔离** + **IM 机器人接入**

---

## 10. 开放问题（评审确认）

1. ~~Embedding/LLM 具体用哪家云端~~ **已定**：Embedding=`BAAI/bge-base-zh-v1.5` 本地服务（768 维，索引建库后不可随意变更）；LLM 生成支持 DeepSeek/OpenAI-compatible；稀疏路首版用 LlamaIndex `BM25Retriever`
2. 首版部署目标机器（内网服务器 IP/资源）—— 影响 Docker 部署形态
3. Golden Set 的标注人力来源 —— 评测质量依赖
