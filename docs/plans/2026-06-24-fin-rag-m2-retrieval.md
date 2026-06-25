# fin-rag M2 实现计划：混合检索 + 重排

> **已废弃（2026-06-25）**：本计划基于手写 Qdrant/RRF 检索，与当前 LlamaIndex 技术方案不兼容，请勿执行。后续按 `docs/design/2026-06-24-fin-rag-design.md` 重新拆分实施计划。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 M0-M1 索引基础上，实现**混合检索**（智谱 Embedding-3 稠密 + Qdrant 服务端 BM25 稀疏，RRF 融合），叠加查询改写、父子回灌、业务加权重排，给定一个 query 能返回带溯源元数据的排序切块列表（`RankedChunk`）。

**Architecture:** `Retriever` 编排：`QueryRewriter(术语归一化) → HybridRetriever(dense+BM25 RRF + domain 过滤) → ParentContextEnricher(回灌父章节) → BusinessReranker(按 doc_type/domain 加权)`。向量库从 M0-M1 的 nameless dense 升级为 **named dense + BM25 sparse**，复用同一 Qdrant 实例。本阶段不含 LLM 问答（属 M3）。

**Tech Stack:** 复用 M0-M1 全部依赖；新增 `qdrant-client` 的 `query_points` / `SparseVectorParams` / `Document(model="qdrant/bm25")` / `FusionQuery(RRF)` 能力。无新增第三方包。

**关键事实（已核对 Qdrant 官方文档 `text-search` / `hybrid-queries`）：**
- `qdrant/bm25` 模型的推理在**自托管 Docker Qdrant 上即可用**（"Inference is only available on Qdrant Cloud, **with the exception of the BM25 model**"）——无需 Qdrant Cloud、无需 FastEmbed。
- BM25 用 `multilingual` tokenizer 原生支持中文（"supports multiple languages, including those with non-Latin alphabets and non-space delimiters"）；配 `language=none` 关闭英文 stemmer/stopword，避免误伤中文。
- BM25 sparse vector 必须配 `modifier=IDF`。
- 混合检索：`query_points(prefetch=[dense 向量路, BM25 文本路], query=FusionQuery(RRF))`。**dense 路用客户端算好的向量**（我们用智谱 Embedding-3），**BM25 路用 `Document(text, model="qdrant/bm25", options=...)`** 让 Qdrant 服务端算。

**与 M0-M1 的衔接（重要）：**
- M0-M1 的 `QdrantVectorStore` 建的是 **nameless dense** 集合；M2 需要 **named dense(`dense`) + sparse(`text`) BM25**。
- 本阶段 `HybridVectorStore.ensure_hybrid_collection()` 会**检测旧 schema 并删除重建**（检测到集合无 `text` sparse 配置即重建）。因 `node_id = {project}:{file_hash}:{chunk_index}` 天然幂等，重建后**重新跑一次 `fin-rag index`** 即可正确回填双向量。
- 接口约定沿用 M0-M1：`Node`（含 `node_id`/`parent_node_id` 等）、`Settings`、`OpenAIEmbedder`。

**重排范围说明：** 本阶段实现 spec §6.2.3 的**业务加权重排**（按 `doc_type`/`domain` 命中提权，纯客户端、无外部模型）。spec §6.1 流程图里的 `BGE-Reranker-v2-m3` 神经精排**留作 M4 评测后的可选增强**（若业务加权后评测不达标再上）。二者解耦，互不影响。

---

## Task 1: HybridVectorStore（named dense + Qdrant BM25 sparse）

**Files:**
- Create: `src/fin_rag/store/hybrid_store.py`
- Test: `tests/test_hybrid_store.py`

- [ ] **Step 1: 写失败测试 `tests/test_hybrid_store.py`**

```python
from unittest.mock import MagicMock

from fin_rag.config import Settings
from fin_rag.models import make_node
from fin_rag.store.hybrid_store import (
    DENSE_NAME,
    SPARSE_NAME,
    BM25_OPTIONS,
    HybridVectorStore,
)


def _settings():
    return Settings(zhipu_api_key="test", qdrant_collection="fin_rag", embedding_dim=1024)


def _node(i):
    return make_node(
        source="fin-online", rel_path="docs/a.md", text=f"账期切换逻辑{i}",
        chunk_index=i, project="fin-online", domain="fin-online", source_type="doc",
        doc_type="business_logic", heading_path="h", file_hash="hash", mtime=0.0,
    )


def test_ensure_rebuilds_when_sparse_missing():
    store = HybridVectorStore(_settings())
    fake = MagicMock()
    fake.collection_exists.return_value = True
    # 旧集合：没有 sparse 配置
    cfg = MagicMock()
    cfg.params.sparse_vectors = {}
    fake.get_collection.return_value.config = cfg
    store._client = fake

    store.ensure_hybrid_collection()

    fake.delete_collection.assert_called_once()
    fake.create_collection.assert_called_once()
    _, kwargs = fake.create_collection.call_args
    assert DENSE_NAME in kwargs["vectors_config"]
    assert SPARSE_NAME in kwargs["sparse_vectors_config"]


def test_ensure_keeps_existing_hybrid_collection():
    store = HybridVectorStore(_settings())
    fake = MagicMock()
    fake.collection_exists.return_value = True
    cfg = MagicMock()
    cfg.params.sparse_vectors = {SPARSE_NAME: MagicMock()}  # 已有 sparse
    fake.get_collection.return_value.config = cfg
    store._client = fake

    store.ensure_hybrid_collection()

    fake.delete_collection.assert_not_called()
    fake.create_collection.assert_not_called()


def test_upsert_hybrid_writes_dense_and_bm25_document():
    store = HybridVectorStore(_settings())
    fake = MagicMock()
    store._client = fake
    nodes = [_node(0), _node(1)]
    dense = [[0.1] * 1024, [0.2] * 1024]
    store.upsert_hybrid(nodes, dense)

    fake.upsert.assert_called_once()
    _, kwargs = fake.upsert.call_args
    point = kwargs["points"][0]
    assert point.id == nodes[0].node_id
    # dense 是数值向量
    assert point.vector[DENSE_NAME] == [0.1] * 1024
    # sparse 是 BM25 Document
    doc = point.vector[SPARSE_NAME]
    assert doc.model == "qdrant/bm25"
    assert doc.options == BM25_OPTIONS


def test_query_hybrid_builds_rrf_with_dense_and_bm25_prefetch():
    store = HybridVectorStore(_settings())
    fake = MagicMock()
    fake.query_points.return_value = MagicMock(points=[])
    store._client = fake

    store.query_hybrid(dense_vector=[0.1] * 1024, text="300007 账期", top_k=10)

    fake.query_points.assert_called_once()
    _, kwargs = fake.query_points.call_args
    prefetch = kwargs["prefetch"]
    assert prefetch[0].using == DENSE_NAME           # dense 向量路
    assert prefetch[0].query == [0.1] * 1024
    assert prefetch[1].using == SPARSE_NAME           # BM25 文本路
    assert prefetch[1].query.model == "qdrant/bm25"
    # 顶层融合
    assert kwargs["query"].fusion is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_hybrid_store.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'fin_rag.store.hybrid_store'`）

- [ ] **Step 3: 写 `src/fin_rag/store/hybrid_store.py`**

```python
"""Hybrid 向量库：named dense（智谱 Embedding-3）+ Qdrant 服务端 BM25 sparse，RRF 混合检索。"""

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from fin_rag.config import Settings
from fin_rag.models import Node

DENSE_NAME = "dense"
SPARSE_NAME = "text"
# BM25 选项：multilingual tokenizer 原生支持中文；language=none 关闭英文 stemmer/stopword
BM25_OPTIONS = {"tokenizer": "multilingual", "language": "none"}
BM25_MODEL = "qdrant/bm25"


class HybridVectorStore:
    """带 BM25 sparse 的 hybrid 存储；自动从 M0-M1 nameless dense 升级（删旧重建）。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = QdrantClient(url=settings.qdrant_url)

    def ensure_hybrid_collection(self) -> None:
        """确保集合为 named dense + sparse(BM25)；旧 schema 则删除重建。"""
        name = self.settings.qdrant_collection
        if self._client.collection_exists(name):
            existing = self._client.get_collection(name).config
            sparse = existing.params.sparse_vectors or {}
            if SPARSE_NAME in sparse:
                return  # 已是 hybrid，保留
            self._client.delete_collection(name)  # nameless dense → 重建
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_NAME: models.VectorParams(
                    size=self.settings.embedding_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def upsert_hybrid(self, nodes: list[Node], dense_vectors: list[list[float]]) -> None:
        """同时写入 dense 向量与 BM25 Document(sparse 由 Qdrant 服务端算)。"""
        points = [
            models.PointStruct(
                id=node.node_id,
                vector={
                    DENSE_NAME: vec,
                    SPARSE_NAME: models.Document(
                        text=node.text, model=BM25_MODEL, options=BM25_OPTIONS
                    ),
                },
                payload=self._payload(node),
            )
            for node, vec in zip(nodes, dense_vectors)
        ]
        self._client.upsert(collection_name=self.settings.qdrant_collection, points=points)

    def query_hybrid(
        self,
        dense_vector: list[float],
        text: str,
        top_k: int,
        domains: Optional[list[str]] = None,
    ):
        """dense + BM25 RRF 融合检索；可选 domain 过滤。"""
        prefetch_limit = top_k * 2
        result = self._client.query_points(
            collection_name=self.settings.qdrant_collection,
            prefetch=[
                models.Prefetch(query=dense_vector, using=DENSE_NAME, limit=prefetch_limit),
                models.Prefetch(
                    query=models.Document(text=text, model=BM25_MODEL, options=BM25_OPTIONS),
                    using=SPARSE_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=self._domain_filter(domains),
        )
        return result.points

    def scroll_points(self, ids: list[str]) -> dict:
        """按 point id 批量取 payload（父子回灌用）。"""
        if not ids:
            return {}
        records = self._client.retrieve(
            collection_name=self.settings.qdrant_collection, ids=ids, with_payload=True
        )
        return {str(r.id): r for r in records}

    def _domain_filter(self, domains: Optional[list[str]]):
        if not domains:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="domain", match=models.MatchAny(any=domains))]
        )

    def _payload(self, node: Node) -> dict:
        return {
            "text": node.text,
            "project": node.project,
            "domain": node.domain,
            "source_type": node.source_type,
            "doc_type": node.doc_type,
            "product_code": node.product_code,
            "channel": node.channel,
            "heading_path": node.heading_path,
            "file_path": node.rel_path,
            "file_hash": node.file_hash,
            "chunk_index": node.chunk_index,
            "parent_node_id": node.parent_node_id,
        }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_hybrid_store.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/store/hybrid_store.py tests/test_hybrid_store.py
git commit -m "feat: HybridVectorStore（named dense + Qdrant BM25 sparse，RRF 检索，schema 自动升级）"
```

---

## Task 2: RankedChunk 不可变模型

**Files:**
- Create: `src/fin_rag/retrieval/__init__.py`
- Create: `src/fin_rag/retrieval/types.py`
- Test: `tests/test_ranked_chunk.py`

- [ ] **Step 1: 写失败测试 `tests/test_ranked_chunk.py`**

```python
import pytest

from fin_rag.retrieval.types import RankedChunk


def _chunk(**over):
    base = dict(
        node_id="fin-online:h:0", text="账期切换", score=0.8, project="fin-online",
        domain="fin-online", source_type="cursor_memory", doc_type="business_logic",
        product_code=None, channel=None, heading_path="账期", file_path="docs/a.md",
        parent_node_id=None,
    )
    base.update(over)
    return RankedChunk(**base)


def test_ranked_chunk_is_immutable():
    c = _chunk()
    with pytest.raises(Exception):
        c.score = 0.1


def test_ranked_chunk_citation_label():
    c = _chunk()
    assert "[fin-online" in c.citation_label()
    assert "docs/a.md" in c.citation_label()
    assert "账期" in c.citation_label()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_ranked_chunk.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/retrieval/__init__.py`（空）**

```python
```

- [ ] **Step 4: 写 `src/fin_rag/retrieval/types.py`**

```python
"""检索结果不可变模型：带溯源标签，父子回灌填充 parent_text。"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RankedChunk:
    """检索返回的单个切块 + 富元数据 + 溯源。"""

    node_id: str
    text: str
    score: float
    project: str
    domain: str
    source_type: str
    doc_type: str
    product_code: Optional[str]
    channel: Optional[str]
    heading_path: str
    file_path: str
    parent_node_id: Optional[str]
    parent_text: Optional[str] = None  # 父子回灌填充

    def citation_label(self) -> str:
        """生成溯源标签：[domain | 文件路径 | §章节]。"""
        return f"[{self.domain} | {self.file_path} | §{self.heading_path}]"
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_ranked_chunk.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: Commit**

```bash
git add src/fin_rag/retrieval/__init__.py src/fin_rag/retrieval/types.py tests/test_ranked_chunk.py
git commit -m "feat: RankedChunk 不可变检索结果模型 + 溯源标签"
```

---

## Task 3: QueryRewriter（术语归一化，单查询）

> **首版只做术语归一化**（口语↔文档术语映射），不做多查询展开。多查询展开二期再加。

**Files:**
- Create: `src/fin_rag/retrieval/query_rewrite.py`
- Test: `tests/test_query_rewrite.py`

- [ ] **Step 1: 写失败测试 `tests/test_query_rewrite.py`**

```python
from fin_rag.retrieval.query_rewrite import QueryRewriter


def test_rewrite_normalizes_whitespace():
    assert QueryRewriter().rewrite("   账期切换怎么处理   ") == "账期切换怎么处理"


def test_rewrite_maps_known_synonym_to_primary_term():
    # 用户口语"结算周期" → 归一化为文档术语"账期"，确保与文档词项一致
    assert "账期" in QueryRewriter().rewrite("结算周期切换怎么处理")


def test_rewrite_no_synonym_passes_through():
    assert QueryRewriter().rewrite("随便一个问题") == "随便一个问题"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_query_rewrite.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/retrieval/query_rewrite.py`**

```python
"""查询改写：中文术语归一化（用户口语 ↔ 文档术语）。

首版策略：把同义词归一化到「主术语」后返回单个查询字符串。
多查询展开（同问题多角度重写、分别检索）二期再加。
"""

# 同义词 → 主术语（把口语归一到文档里用的词，提升 BM25/dense 命中）
DEFAULT_TERM_MAP: dict[str, str] = {
    "结算周期": "账期",
    "billing cycle": "账期",
    "降档": "降查得",
    "degrade": "降查得",
    "产品ID": "PID",
    "产品标识": "PID",
}


class QueryRewriter:
    def __init__(self, term_map: dict[str, str] | None = None) -> None:
        self.term_map = term_map if term_map is not None else DEFAULT_TERM_MAP

    def rewrite(self, query: str) -> str:
        text = " ".join(query.split()).strip()
        for synonym, primary in self.term_map.items():
            if synonym in text:
                text = text.replace(synonym, primary)
        return text
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_query_rewrite.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/retrieval/query_rewrite.py tests/test_query_rewrite.py
git commit -m "feat: QueryRewriter（中文术语归一化，单查询）"
```

---

## Task 4: HybridRetriever（dense + BM25 RRF）

**Files:**
- Create: `src/fin_rag/retrieval/hybrid.py`
- Test: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: 写失败测试 `tests/test_hybrid_retriever.py`**

```python
from unittest.mock import MagicMock

from fin_rag.retrieval.hybrid import HybridRetriever


def _point(node_id, score, **pl):
    p = MagicMock()
    p.id = node_id
    p.score = score
    p.payload = pl
    return p


def test_retrieve_embeds_query_then_maps_to_ranked_chunk():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.2] * 1024]
    store = MagicMock()
    store.query_hybrid.return_value = [
        _point("fin-online:h:0", 0.9, text="账期", domain="fin-online",
               doc_type="business_logic", heading_path="账期", file_path="docs/a.md")
    ]
    ret = HybridRetriever(store=store, embedder=embedder, top_k=10)

    chunks = ret.retrieve("账期切换")

    embedder.embed.assert_called_once_with(["账期切换"])
    store.query_hybrid.assert_called_once()
    _, kw = store.query_hybrid.call_args
    assert kw["dense_vector"] == [0.2] * 1024
    assert kw["text"] == "账期切换"
    assert kw["top_k"] == 10
    assert len(chunks) == 1
    assert chunks[0].node_id == "fin-online:h:0"
    assert chunks[0].score == 0.9
    assert chunks[0].domain == "fin-online"
    assert chunks[0].citation_label().startswith("[fin-online")


def test_retrieve_passes_domains_filter():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1]]
    store = MagicMock()
    store.query_hybrid.return_value = []
    ret = HybridRetriever(store=store, embedder=embedder, top_k=5)

    ret.retrieve("x", domains=["opdata"])

    _, kw = store.query_hybrid.call_args
    assert kw["domains"] == ["opdata"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/retrieval/hybrid.py`**

```python
"""混合检索：智谱 Embedding-3 稠密 + Qdrant BM25 稀疏，RRF 融合，domain 过滤。"""

from typing import Optional

from fin_rag.pipeline.embedder import OpenAIEmbedder
from fin_rag.retrieval.types import RankedChunk
from fin_rag.store.hybrid_store import HybridVectorStore


class HybridRetriever:
    def __init__(
        self,
        store: HybridVectorStore,
        embedder: OpenAIEmbedder,
        top_k: int = 10,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self, query: str, domains: Optional[list[str]] = None
    ) -> list[RankedChunk]:
        dense_vector = self.embedder.embed([query])[0]
        points = self.store.query_hybrid(
            dense_vector=dense_vector,
            text=query,
            top_k=self.top_k,
            domains=domains,
        )
        return [self._to_chunk(p) for p in points]

    def _to_chunk(self, point) -> RankedChunk:
        pl = point.payload or {}
        return RankedChunk(
            node_id=str(point.id),
            text=pl.get("text", ""),
            score=float(point.score),
            project=pl.get("project", ""),
            domain=pl.get("domain", ""),
            source_type=pl.get("source_type", ""),
            doc_type=pl.get("doc_type", ""),
            product_code=pl.get("product_code"),
            channel=pl.get("channel"),
            heading_path=pl.get("heading_path", ""),
            file_path=pl.get("file_path", ""),
            parent_node_id=pl.get("parent_node_id"),
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/retrieval/hybrid.py tests/test_hybrid_retriever.py
git commit -m "feat: HybridRetriever（dense+BM25 RRF 混合检索 → RankedChunk）"
```

---

## Task 5: ParentContextEnricher（父子回灌）

**Files:**
- Create: `src/fin_rag/retrieval/parent_context.py`
- Test: `tests/test_parent_context.py`

- [ ] **Step 1: 写失败测试 `tests/test_parent_context.py`**

```python
from unittest.mock import MagicMock

from fin_rag.retrieval.parent_context import ParentContextEnricher
from fin_rag.retrieval.types import RankedChunk


def _chunk(**over):
    base = dict(
        node_id="c0", text="子块", score=0.8, project="fin-online", domain="fin-online",
        source_type="doc", doc_type="tech_plan", product_code=None, channel=None,
        heading_path="父/子", file_path="docs/a.md", parent_node_id=None,
    )
    base.update(over)
    return RankedChunk(**base)


def test_enrich_fills_parent_text_from_store():
    parent = MagicMock()
    parent.payload = {"text": "父章节正文"}
    store = MagicMock()
    store.scroll_points.return_value = {"p1": parent}
    enricher = ParentContextEnricher(store)

    out = enricher.enrich([_chunk(parent_node_id="p1")])

    store.scroll_points.assert_called_once_with(["p1"])
    assert out[0].parent_text == "父章节正文"
    assert out[0].text == "子块"  # 正文不被覆盖


def test_enrich_skips_when_no_parent():
    store = MagicMock()
    store.scroll_points.return_value = {}
    enricher = ParentContextEnricher(store)

    out = enricher.enrich([_chunk(parent_node_id=None)])

    store.scroll_points.assert_called_once_with([])  # 无 id → 空列表
    assert out[0].parent_text is None


def test_enrich_dedups_parent_ids():
    store = MagicMock()
    store.scroll_points.return_value = {"p1": MagicMock(payload={"text": "父"})}
    enricher = ParentContextEnricher(store)

    enricher.enrich([_chunk(parent_node_id="p1"), _chunk(node_id="c1", parent_node_id="p1")])

    store.scroll_points.assert_called_once_with(["p1"])  # 去重
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_parent_context.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/retrieval/parent_context.py`**

```python
"""父子回灌：命中子块 → 回灌父章节文本，补全长业务逻辑上下文（承接 M0-M1 §5.3）。"""

from dataclasses import replace

from fin_rag.retrieval.types import RankedChunk
from fin_rag.store.hybrid_store import HybridVectorStore


class ParentContextEnricher:
    def __init__(self, store: HybridVectorStore) -> None:
        self.store = store

    def enrich(self, chunks: list[RankedChunk]) -> list[RankedChunk]:
        parent_ids = list({c.parent_node_id for c in chunks if c.parent_node_id})
        parents = self.store.scroll_points(parent_ids)
        return [self._with_parent(c, parents) for c in chunks]

    def _with_parent(self, chunk: RankedChunk, parents: dict) -> RankedChunk:
        if not chunk.parent_node_id or chunk.parent_node_id not in parents:
            return chunk
        parent_text = (parents[chunk.parent_node_id].payload or {}).get("text", "")
        return replace(chunk, parent_text=parent_text)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_parent_context.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/retrieval/parent_context.py tests/test_parent_context.py
git commit -m "feat: ParentContextEnricher（父子回灌，补全长业务逻辑上下文）"
```

---

## Task 6: BusinessReranker（业务加权重排，无外部模型）

**Files:**
- Create: `src/fin_rag/retrieval/business_rerank.py`
- Test: `tests/test_business_rerank.py`

- [ ] **Step 1: 写失败测试 `tests/test_business_rerank.py`**

```python
from fin_rag.retrieval.business_rerank import BusinessReranker
from fin_rag.retrieval.types import RankedChunk


def _chunk(node_id, score, doc_type="tech_plan", domain="fin-online"):
    return RankedChunk(
        node_id=node_id, text="t", score=score, project="fin-online", domain=domain,
        source_type="doc", doc_type=doc_type, product_code=None, channel=None,
        heading_path="h", file_path="docs/a.md", parent_node_id=None,
    )


def test_rerank_promotes_business_logic_by_doc_type():
    reranker = BusinessReranker()
    chunks = [
        _chunk("low", 0.50, doc_type="tech_plan"),
        _chunk("high", 0.49, doc_type="business_logic"),  # 分数低但 doc_type 提权后更高
    ]
    out = reranker.rerank(chunks)
    assert out[0].node_id == "high"
    assert out[0].score > out[1].score


def test_rerank_keeps_order_when_scores_clear():
    reranker = BusinessReranker()
    chunks = [_chunk("a", 0.99), _chunk("b", 0.10)]
    out = reranker.rerank(chunks)
    assert [c.node_id for c in out] == ["a", "b"]


def test_rerank_applies_domain_bonus_when_filtered():
    reranker = BusinessReranker()
    chunks = [
        _chunk("other", 0.50, domain="fin-online"),
        _chunk("hit", 0.49, domain="opdata"),
    ]
    out = reranker.rerank(chunks, domains=["opdata"])
    assert out[0].node_id == "hit"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_business_rerank.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/retrieval/business_rerank.py`**

```python
"""业务加权重排：对稀缺 doc_type 与 domain 命中提权（纯客户端，无外部模型，对齐 spec §6.2.3）。"""

from dataclasses import replace
from typing import Optional

from fin_rag.retrieval.types import RankedChunk

# 稀缺/权威内容提权权重（business_logic=memory/special，engineering_rule=.cursor/rules）
DEFAULT_BOOST: dict[str, float] = {
    "business_logic": 0.15,
    "engineering_rule": 0.12,
    "spec": 0.05,
}
DEFAULT_DOMAIN_BONUS = 0.05


class BusinessReranker:
    def __init__(
        self,
        boost_table: Optional[dict[str, float]] = None,
        domain_bonus: float = DEFAULT_DOMAIN_BONUS,
    ) -> None:
        self.boost_table = boost_table if boost_table is not None else DEFAULT_BOOST
        self.domain_bonus = domain_bonus

    def rerank(
        self, chunks: list[RankedChunk], domains: Optional[list[str]] = None
    ) -> list[RankedChunk]:
        target = set(domains or [])
        scored = [(c, self._adjusted_score(c, target)) for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [replace(c, score=s) for c, s in scored]

    def _adjusted_score(self, chunk: RankedChunk, target: set[str]) -> float:
        score = chunk.score
        score += self.boost_table.get(chunk.doc_type, 0.0)
        if target and chunk.domain in target:
            score += self.domain_bonus
        return score
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_business_rerank.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/retrieval/business_rerank.py tests/test_business_rerank.py
git commit -m "feat: BusinessReranker（按 doc_type/domain 业务加权重排）"
```

---

## Task 7: Retriever 编排 + 检索结果合并 + CLI search + 端到端验证

**Files:**
- Create: `src/fin_rag/retrieval/retriever.py`
- Modify: `src/fin_rag/cli.py`（新增 `search` 子命令）
- Modify: `config/settings.yaml`（新增 `retrieval` 段）
- Modify: `src/fin_rag/config.py`（`Settings` 增 `retrieval_top_k`）
- Test: `tests/test_retriever.py`

- [ ] **Step 1: 写失败测试 `tests/test_retriever.py`（组合各组件，mock 底层）**

```python
from unittest.mock import MagicMock

from fin_rag.retrieval.retriever import Retriever
from fin_rag.retrieval.types import RankedChunk


def _chunk(node_id, score, doc_type="business_logic", domain="fin-online"):
    return RankedChunk(
        node_id=node_id, text="账期", score=score, project="fin-online", domain=domain,
        source_type="cursor_memory", doc_type=doc_type, product_code=None, channel=None,
        heading_path="账期", file_path="docs/a.md", parent_node_id=None,
    )


def test_search_runs_full_pipeline():
    hybrid = MagicMock()
    hybrid.retrieve.return_value = [_chunk("c0", 0.8), _chunk("c1", 0.3)]
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "账期"   # 归一化后返回单个查询字符串
    parent = MagicMock()
    parent.enrich.side_effect = lambda xs: xs
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda xs, domains=None: xs

    r = Retriever(hybrid=hybrid, rewriter=rewriter, parent=parent, reranker=reranker)
    out = r.search("账期切换")

    rewriter.rewrite.assert_called_once_with("账期切换")
    hybrid.retrieve.assert_called_once_with("账期", domains=None)
    parent.enrich.assert_called_once()
    reranker.rerank.assert_called_once()
    assert len(out) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'fin_rag.retrieval.retriever'`）

- [ ] **Step 3: 写 `src/fin_rag/retrieval/retriever.py`**

```python
"""检索编排：查询改写（单查询）→ 混合检索 → 父子回灌 → 业务加权重排。"""

from typing import Optional

from fin_rag.retrieval.business_rerank import BusinessReranker
from fin_rag.retrieval.hybrid import HybridRetriever
from fin_rag.retrieval.parent_context import ParentContextEnricher
from fin_rag.retrieval.query_rewrite import QueryRewriter
from fin_rag.retrieval.types import RankedChunk


class Retriever:
    def __init__(
        self,
        hybrid: HybridRetriever,
        rewriter: Optional[QueryRewriter] = None,
        parent: Optional[ParentContextEnricher] = None,
        reranker: Optional[BusinessReranker] = None,
    ) -> None:
        self.hybrid = hybrid
        self.rewriter = rewriter
        self.parent = parent
        self.reranker = reranker

    def search(
        self,
        query: str,
        domains: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[RankedChunk]:
        if top_k is not None:
            self.hybrid.top_k = top_k
        q = self.rewriter.rewrite(query) if self.rewriter is not None else query
        chunks = self.hybrid.retrieve(q, domains=domains)
        if self.parent is not None:
            chunks = self.parent.enrich(chunks)
        if self.reranker is not None:
            chunks = self.reranker.rerank(chunks, domains=domains)
        return chunks
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 改 `config/settings.yaml` 增检索配置**

在 `qdrant` 段后追加：

```yaml
retrieval:
  top_k: 10
  score_gate: 0.3   # 召回门控阈值（M3 防幻觉闸①使用）
```

- [ ] **Step 6: 改 `src/fin_rag/config.py`（`Settings` 增字段 + `load_settings` 回填）**

在 `Settings` 类内 `qdrant_collection` 字段之后追加：

```python
    # 检索（M2）
    retrieval_top_k: int = 10
    retrieval_multi_query: bool = False
    retrieval_score_gate: float = 0.3
```

在 `load_settings` 的 `return Settings(...)` 内追加（读取 `retrieval` 段）：

```python
        retrieval_top_k=retrieval.get("top_k", 10),
        retrieval_multi_query=retrieval.get("multi_query", False),
        retrieval_score_gate=retrieval.get("score_gate", 0.3),
```

并在 `load_settings` 函数体里 `embedding = ...` 那一段附近加：

```python
    retrieval = raw.get("retrieval", {})
```

- [ ] **Step 7: 改 `src/fin_rag/cli.py`（新增 `search` 子命令 + 工厂）**

在文件顶部 import 区追加（注意：`OpenAIEmbedder` 已在 M0-M1 的 `cli.py` import 过，此处**不要重复**）：

```python
from fin_rag.retrieval.business_rerank import BusinessReranker
from fin_rag.retrieval.hybrid import HybridRetriever
from fin_rag.retrieval.parent_context import ParentContextEnricher
from fin_rag.retrieval.query_rewrite import QueryRewriter
from fin_rag.retrieval.retriever import Retriever
from fin_rag.store.hybrid_store import HybridVectorStore
```

在 `_build_pipeline()` 函数之后新增工厂：

```python
def _build_retriever():
    settings = load_settings(CONFIG_DIR / "settings.yaml")
    store = HybridVectorStore(settings)
    hybrid = HybridRetriever(
        store=store, embedder=OpenAIEmbedder(settings), top_k=settings.retrieval_top_k
    )
    return Retriever(
        hybrid=hybrid,
        rewriter=QueryRewriter(),
        parent=ParentContextEnricher(store),
        reranker=BusinessReranker(),
        multi_query=settings.retrieval_multi_query,
    )
```

在 `main()` 的 `sub.add_parser("index", ...)` 之后新增 `search` 子命令与处理分支：

```python
    search_p = sub.add_parser("search", help="混合检索测试")
    search_p.add_argument("query", help="查询文本")
    search_p.add_argument("--domain", action="append", default=None, help="按 domain 筛选（可多次）")
    search_p.add_argument("--top-k", type=int, default=None, help="返回数量")
```

在 `if args.cmd == "index":` 分支之后追加：

```python
    if args.cmd == "search":
        try:
            chunks = _build_retriever().search(
                args.query, domains=args.domain, top_k=args.top_k
            )
            for i, c in enumerate(chunks, 1):
                log.info("#%d score=%.4f %s", i, c.score, c.citation_label())
                log.info("    %s", c.text[:120].replace("\n", " "))
            log.info("命中 %d 条", len(chunks))
            return 0
        except Exception as exc:
            log.error("检索失败：%s", exc)
            return 1
```

- [ ] **Step 8: 运行全量测试 + 覆盖率**

Run: `pytest -v`
Expected: 全部 PASS，`fin_rag` 覆盖率 ≥ 80%

- [ ] **Step 9: 端到端手动验证（需 Qdrant + 真实 API key + M2 重建后重新索引）**

```bash
cd /Users/wangfei/yulore/fin-rag
docker compose up -d qdrant
source .venv/bin/activate
# M2 schema 升级会自动重建集合，需重新写双向量
fin-rag index
# 验证 BM25（精确产品号/术语）与 dense（语义）都能召回
fin-rag search "300007 的鉴权方式"
fin-rag search "账期切换逻辑"
fin-rag search "日志三级备份" --domain fin-online
```
Expected：
- `fin-rag index` 完成后集合为 named dense + sparse（可在 Dashboard 看到两个向量名）。
- `search` 返回 `RankedChunk` 列表，每条含 `score` + 溯源标签 `[domain | 文件路径 | §章节]`。
- 精确产品号查询（如 `300007`）BM25 路应精确命中；语义查询（"账期切换逻辑"）dense 路应召回 `memory/special` 业务逻辑。

用以下命令抽样核验集合 schema：
```bash
python -c "from qdrant_client import QdrantClient; c=QdrantClient(url='http://localhost:6333'); cfg=c.get_collection('fin_rag').config.params; print('dense' in cfg.vectors, list(cfg.sparse_vectors))"
```
Expected: `True ['text']`

- [ ] **Step 10: Commit**

```bash
git add src/fin_rag/retrieval/retriever.py src/fin_rag/cli.py config/settings.yaml src/fin_rag/config.py tests/test_retriever.py
git commit -m "feat: Retriever 编排 + CLI search（改写→混合检索→父子回灌→业务加权，端到端可验证）"
```

---

## Task 8: 统一索引写入为 Hybrid（dense + BM25 双向量）

> **修正（self-review 发现的跨阶段一致性问题）**：M0-M1 的 `IndexPipeline` 写 nameless-dense（`QdrantVectorStore.upsert`），而 M2 检索要 named dense + BM25 sparse。若不统一，`fin-rag index` 跑出的集合与 `search/ask` 的 hybrid schema 不匹配。本任务把索引写入切到 `HybridVectorStore`，使 index 与 search/ask 共用同一 hybrid schema。

**Files:**
- Modify: `src/fin_rag/pipeline/pipeline.py`（`IndexPipeline.run` 改调 `ensure_hybrid_collection` + `upsert_hybrid`）
- Modify: `src/fin_rag/cli.py`（`_build_pipeline` 的 store 换 `HybridVectorStore`）
- Modify: `tests/test_pipeline.py`（store mock 改 `upsert_hybrid`）

- [ ] **Step 1: 更新 `tests/test_pipeline.py`（store mock 用 hybrid 接口）**

把 `test_pipeline_runs_end_to_end_with_mocked_embed_and_store` 末尾的断言替换（方法名 `ensure_collection` → `ensure_hybrid_collection`，`upsert` → `upsert_hybrid`）：

```python
    count = pipe.run()

    assert count > 0
    store.ensure_hybrid_collection.assert_called_once()
    store.upsert_hybrid.assert_called()
    # 传给 store 的 node payload 含正确 domain
    _, kwargs = store.upsert_hybrid.call_args
    payloads = [p.payload for p in kwargs["points"]]
    assert all(p["domain"] == "sample" for p in payloads)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL（`upsert_hybrid` 未被调用 / `ensure_collection` 仍被调）

- [ ] **Step 3: 改 `src/fin_rag/pipeline/pipeline.py` 的 `IndexPipeline.run`**

两处替换（方法体其余不变）：
- `self.store.ensure_collection()` → `self.store.ensure_hybrid_collection()`
- `self.store.upsert(nodes, vectors)` → `self.store.upsert_hybrid(nodes, vectors)`

- [ ] **Step 4: 改 `src/fin_rag/cli.py` 的 `_build_pipeline`**

import 区加（`HybridVectorStore` 尚未在 cli import）：

```python
from fin_rag.store.hybrid_store import HybridVectorStore
```

把 `store=QdrantVectorStore(settings)` 改为 `store=HybridVectorStore(settings)`。

- [ ] **Step 5: 运行全量测试确认通过**

Run: `pytest -v`
Expected: 全部 PASS

- [ ] **Step 6: 端到端再验证**

```bash
fin-rag index    # 现在写 named dense + BM25 sparse 双向量
fin-rag search "300007"
```
Expected：`index` 后集合为 hybrid（Dashboard 可见 `dense` + `text` 两个向量名）；`search` 正常召回。

- [ ] **Step 7: Commit**

```bash
git add src/fin_rag/pipeline/pipeline.py src/fin_rag/cli.py tests/test_pipeline.py
git commit -m "fix: 索引写入统一为 Hybrid（IndexPipeline→HybridVectorStore，对齐检索 schema）"
```

---

## 完成标准（M2）

- [ ] 全部 7 个任务测试通过，`pytest` 覆盖率 ≥ 80%
- [ ] `fin-rag index` 在 M2 schema 下重建集合并写入 dense + BM25 sparse 双向量
- [ ] `fin-rag search` 能端到端返回排序 `RankedChunk`，含溯源标签
- [ ] 精确查询（产品号/PID）靠 BM25 命中、语义查询靠 dense 命中，RRF 融合
- [ ] domain 筛选、父子回灌、业务加权三条链路各自有单测覆盖

## 下一步

M2 完成、检索验证通过后，进入**第三份计划（M3）**：防幻觉三道闸（检索门控 + 生成硬约束 + 引用事后校验）+ 溯源 + DeepSeek 问答编排（QAChain），把 `Retriever` 的输出接入 LLM 生成带引用的答案。

---

## 附录：关键 Qdrant API 参考（已核对官方文档）

**集合（named dense + BM25 sparse）：**
```python
client.create_collection(
    collection_name="fin_rag",
    vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
    sparse_vectors_config={"text": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
```

**入库（dense 用客户端向量，sparse 用 BM25 Document）：**
```python
models.PointStruct(
    id=node_id,
    vector={
        "dense": dense_vec,
        "text": models.Document(text=..., model="qdrant/bm25",
                                options={"tokenizer": "multilingual", "language": "none"}),
    },
    payload={...},
)
```

**混合查询（RRF 融合）：**
```python
client.query_points(
    collection_name="fin_rag",
    prefetch=[
        models.Prefetch(query=dense_vec, using="dense", limit=20),
        models.Prefetch(query=models.Document(text=q, model="qdrant/bm25",
                                              options={"tokenizer": "multilingual", "language": "none"}),
                        using="text", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
    with_payload=True,
    query_filter=models.Filter(must=[models.FieldCondition(key="domain",
                                                            match=models.MatchAny(any=domains))]),
)
```
