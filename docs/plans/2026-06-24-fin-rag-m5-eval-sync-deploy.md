# fin-rag M5 实现计划：Golden Set 评测 + 增量同步 + Docker 部署

> **已废弃（2026-06-25）**：本计划依赖旧索引、检索和 QA 接口，与当前 LlamaIndex 技术方案不兼容，请勿执行。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 闭环收尾三件事：① **Golden Set 评测**（召回命中率 / 引用准确率 / 拒答准确率 / 忠实度），改检索/切分参数可回归；② **增量同步**（hash+mtime 幂等，复用 `MetadataStore.sync_state`），文档变了只重处理变更文件；③ **Docker 部署**（app + qdrant 一键起，团队内网访问）。

**Architecture:**
- 评测：`eval/loader.py`（金标加载）+ `eval/metrics.py`（指标）+ `eval/run_eval.py`（脚本）。
- 同步：`SyncService` 编排，复用 `IndexPipeline.process_file`（本阶段从 `run` 抽出）+ `HybridVectorStore.delete_by_file_hash`（新增）+ `MetadataStore.sync_state`（M4 已建表）。
- 部署：`Dockerfile` + `docker-compose.yml`（app 依赖 qdrant）。

**Tech Stack:** 标准库 + 复用 M0-M4；无新增第三方包（增量同步首版用 `fin-rag sync` 手动触发；APScheduler/cron 定时留作部署期，避免新依赖）。

**接口约定（沿用）：** M3 `QAChain.ask`/`QAResult`/`Citation`、M2 `HybridVectorStore`/`Retriever`、M4 `MetadataStore`、M0-M1 `IndexPipeline`/`Settings`。

---

## Task 1: Golden Set 加载

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/golden_set.jsonl`
- Create: `eval/loader.py`
- Test: `tests/test_golden_loader.py`

- [ ] **Step 1: 写 `eval/golden_set.jsonl`（种子 6 条；团队后续扩充至 30-50）**

每行一个 case：`expect_files` 用文件路径子串（命中即算召回），`expect_keywords` 是答案应覆盖的要点，`expect_refuse=true` 表示该题知识库无覆盖、应拒答。

```jsonl
{"id":"g1","query":"账期切换逻辑是什么","expect_files":["账期"],"expect_keywords":["账期","切换"],"expect_refuse":false}
{"id":"g2","query":"降查得怎么处理","expect_files":["降查得"],"expect_keywords":["降查得","降档"],"expect_refuse":false}
{"id":"g3","query":"日志三级备份机制","expect_files":["日志"],"expect_keywords":["日志","备份"],"expect_refuse":false}
{"id":"g4","query":"LanChen 渠道怎么对接","expect_files":["channels","LanChen"],"expect_keywords":["LanChen","渠道"],"expect_refuse":false}
{"id":"g5","query":"防幻觉规则要求什么","expect_files":["anti-hallucination"],"expect_keywords":["上下文","引用"],"expect_refuse":false}
{"id":"g6","query":"火星殖民的金融政策","expect_files":[],"expect_keywords":[],"expect_refuse":true}
```

- [ ] **Step 2: 写失败测试 `tests/test_golden_loader.py`**

```python
from pathlib import Path

from eval.loader import GoldenCase, load_golden

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"


def test_load_returns_cases():
    cases = load_golden(GOLDEN)
    assert len(cases) >= 6
    assert all(isinstance(c, GoldenCase) for c in cases)
    g1 = next(c for c in cases if c.id == "g1")
    assert "账期" in g1.query
    assert g1.expect_files and "账期" in g1.expect_files
    assert g1.expect_refuse is False


def test_refuse_case_present():
    cases = load_golden(GOLDEN)
    refuse = [c for c in cases if c.expect_refuse]
    assert len(refuse) >= 1
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_golden_loader.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'eval'`）

- [ ] **Step 4: 写 `eval/__init__.py`（空）**

```python
```

- [ ] **Step 5: 写 `eval/loader.py`**

```python
"""Golden Set 加载：每行一个 JSON case → 不可变 GoldenCase。"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldenCase:
    id: str
    query: str
    expect_files: list[str]
    expect_keywords: list[str]
    expect_refuse: bool


def load_golden(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                GoldenCase(
                    id=obj["id"],
                    query=obj["query"],
                    expect_files=list(obj.get("expect_files", [])),
                    expect_keywords=list(obj.get("expect_keywords", [])),
                    expect_refuse=bool(obj.get("expect_refuse", False)),
                )
            )
    return cases
```

- [ ] **Step 6: 在 `pyproject.toml` 让 `eval` 可被 import + 测试可见**

`[tool.pytest.ini_options]` 的 `pythonpath` 追加根目录（让 `eval` 包与 `src/fin_rag` 都在路径上）：

```toml
pythonpath = ["src", "."]
```

- [ ] **Step 7: 运行测试确认通过**

Run: `pytest tests/test_golden_loader.py -v`
Expected: PASS（2 passed）

- [ ] **Step 8: Commit**

```bash
git add eval/ tests/test_golden_loader.py pyproject.toml
git commit -m "feat: Golden Set 加载（种子 6 条 + 不可变 GoldenCase）"
```

---

## Task 2: 评测指标 + run_eval 脚本

**Files:**
- Create: `eval/metrics.py`
- Create: `eval/run_eval.py`
- Test: `tests/test_eval_metrics.py`

> **忠实度(faithfulness)简化说明**：首版用「答案覆盖 expect_keywords 的比例」近似（无需 LLM-judge，可离线跑）；真实的 LLM-as-judge 留作增强（spec §8.4 标注）。

- [ ] **Step 1: 写失败测试 `tests/test_eval_metrics.py`**

```python
from unittest.mock import MagicMock

from eval.loader import GoldenCase
from eval.metrics import CaseResult, evaluate_case, summarize


def _case(**o):
    base = dict(id="g1", query="账期", expect_files=["账期"], expect_keywords=["账期", "切换"],
                expect_refuse=False)
    base.update(o)
    return GoldenCase(**base)


def _result(files=None, answer="", gated=False, valid=True):
    r = MagicMock()
    r.citations = [MagicMock(file_path=f) for f in (files or [])]
    r.answer = answer
    r.gated = gated
    r.citation_check = MagicMock(valid=valid)
    return r


def test_recall_hit_when_expected_file_present():
    cr = evaluate_case(_case(), _result(files=["docs/账期切换.md"]))
    assert cr.recall_hit is True


def test_recall_miss_when_no_expected_file():
    cr = evaluate_case(_case(expect_files=["不存在"]), _result(files=["docs/其他.md"]))
    assert cr.recall_hit is False


def test_refusal_correctness():
    # 该拒答且拒答 → 对；不该拒答但拒答 → 错
    assert evaluate_case(_case(expect_refuse=True), _result(gated=True)).refusal_correct is True
    assert evaluate_case(_case(expect_refuse=False), _result(gated=True)).refusal_correct is False


def test_keyword_coverage():
    cr = evaluate_case(_case(expect_keywords=["账期", "切换", "降档"]),
                       _result(answer="账期切换的逻辑是…"))  # 覆盖 2/3
    assert cr.keyword_coverage == 2 / 3


def test_summarize_aggregates_rates():
    crs = [
        CaseResult("g1", recall_hit=True, refusal_correct=True, keyword_coverage=1.0, citation_valid=True),
        CaseResult("g2", recall_hit=False, refusal_correct=False, keyword_coverage=0.0, citation_valid=False),
    ]
    rep = summarize(crs)
    assert rep.recall_rate == 0.5
    assert rep.refusal_rate == 0.5
    assert rep.faithfulness == 0.5
    assert rep.citation_accuracy == 0.5
    assert len(rep.details) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_eval_metrics.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `eval/metrics.py`**

```python
"""评测指标：召回命中率 / 拒答准确率 / 忠实度(关键词覆盖近似) / 引用准确率。"""

from dataclasses import dataclass, field

from eval.loader import GoldenCase


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    recall_hit: bool
    refusal_correct: bool
    keyword_coverage: float
    citation_valid: bool


@dataclass(frozen=True)
class EvalReport:
    recall_rate: float
    refusal_rate: float
    faithfulness: float
    citation_accuracy: float
    details: list = field(default_factory=list)


def evaluate_case(case: GoldenCase, result) -> CaseResult:
    retrieved = [c.file_path for c in result.citations]
    if case.expect_files:
        recall = any(any(exp in f for f in retrieved) for exp in case.expect_files)
    else:
        recall = True  # 无期望文件约束（如拒答题）
    refusal = (case.expect_refuse == result.gated)
    if case.expect_keywords:
        hit = sum(1 for k in case.expect_keywords if k in result.answer)
        coverage = hit / len(case.expect_keywords)
    else:
        coverage = 1.0
    cite_valid = result.citation_check.valid if result.citation_check is not None else True
    return CaseResult(case.id, recall, refusal, coverage, cite_valid)


def summarize(results: list[CaseResult]) -> EvalReport:
    n = len(results) or 1
    return EvalReport(
        recall_rate=sum(r.recall_hit for r in results) / n,
        refusal_rate=sum(r.refusal_correct for r in results) / n,
        faithfulness=sum(r.keyword_coverage for r in results) / n,
        citation_accuracy=sum(r.citation_valid for r in results) / n,
        details=list(results),
    )
```

- [ ] **Step 4: 写 `eval/run_eval.py`（脚本：端到端跑 QAChain，输出报告）**

```python
"""评测脚本：对 Golden Set 逐条跑 QAChain，输出指标报告（终端 + JSON）。"""

import json
from pathlib import Path

from eval.loader import load_golden
from eval.metrics import evaluate_case, summarize

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"
OUT = Path(__file__).resolve().parents[1] / "eval" / "last_report.json"


def run(qa_chain, golden_path: Path = GOLDEN) -> dict:
    cases = load_golden(golden_path)
    crs = [evaluate_case(c, qa_chain.ask(c.query)) for c in cases]
    rep = summarize(crs)
    return {
        "recall_rate": rep.recall_rate,
        "refusal_rate": rep.refusal_rate,
        "faithfulness": rep.faithfulness,
        "citation_accuracy": rep.citation_accuracy,
        "details": [
            {"case_id": d.case_id, "recall": d.recall_hit, "refusal": d.refusal_correct,
             "keyword_coverage": d.keyword_coverage, "citation_valid": d.citation_valid}
            for d in rep.details
        ],
    }


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from fin_rag.cli import _build_qa_chain

    report = run(_build_qa_chain())
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("召回命中率   :", round(report["recall_rate"], 3))
    print("拒答准确率   :", round(report["refusal_rate"], 3))
    print("忠实度(关键词):", round(report["faithfulness"], 3))
    print("引用准确率   :", round(report["citation_accuracy"], 3))
    print("明细已写入   :", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_eval_metrics.py -v`
Expected: PASS（5 passed）

- [ ] **Step 6: Commit**

```bash
git add eval/metrics.py eval/run_eval.py tests/test_eval_metrics.py
git commit -m "feat: 评测指标（召回/拒答/忠实度/引用准确率）+ run_eval 脚本"
```

---

## Task 3: 增量同步 SyncService

**Files:**
- Modify: `src/fin_rag/pipeline/pipeline.py`（抽出 `IndexPipeline.process_file`，`run` 复用）
- Modify: `src/fin_rag/store/hybrid_store.py`（加 `delete_by_file_hash`）
- Create: `src/fin_rag/sync/__init__.py`
- Create: `src/fin_rag/sync/sync_service.py`
- Modify: `src/fin_rag/cli.py`（`sync` 子命令）
- Test: `tests/test_sync_service.py`

- [ ] **Step 1: 写失败测试 `tests/test_sync_service.py`**

```python
from unittest.mock import MagicMock

from fin_rag.models import RawFile
from fin_rag.sync.sync_service import SyncService


def _raw(rel, h):
    return RawFile(source="fin-online", rel_path=rel, abs_path="/x/" + rel,
                   mtime=1.0, file_hash=h)


def _pipeline(current_files):
    pipeline = MagicMock()
    ds = MagicMock()
    ds.list_files.return_value = current_files
    pipeline.datasource_factory.return_value = ds
    pipeline.process_file.return_value = 3
    pipeline.sources = [MagicMock(name="fin-online")]
    pipeline.sources[0].name = "fin-online"
    return pipeline


def test_sync_indexes_new_files():
    pipeline = _pipeline([_raw("docs/new.md", "h1")])
    store = MagicMock()
    meta = MagicMock()
    meta.all_sync_state.return_value = {}  # 无历史

    rep = SyncService(pipeline, store, meta).sync()

    pipeline.process_file.assert_called_once()
    meta.set_sync_state.assert_called_once()
    assert rep.added == 1 and rep.updated == 0 and rep.deleted == 0


def test_sync_reindexes_changed_files_and_deletes_old_hash():
    pipeline = _pipeline([_raw("docs/a.md", "hNew")])
    store = MagicMock()
    meta = MagicMock()
    meta.all_sync_state.return_value = {"docs/a.md": {"file_hash": "hOld", "mtime": 0.0}}

    rep = SyncService(pipeline, store, meta).sync()

    store.delete_by_file_hash.assert_called_once_with("fin-online", "hOld")
    pipeline.process_file.assert_called_once()
    assert rep.updated == 1


def test_sync_deletes_removed_files():
    pipeline = _pipeline([])  # 当前无文件
    store = MagicMock()
    meta = MagicMock()
    meta.all_sync_state.return_value = {"docs/gone.md": {"file_hash": "hG", "mtime": 0.0}}

    rep = SyncService(pipeline, store, meta).sync()

    store.delete_by_file_hash.assert_called_once_with("fin-online", "hG")
    meta.delete_sync_state.assert_called_once()
    assert rep.deleted == 1


def test_sync_unchanged_files_do_nothing():
    pipeline = _pipeline([_raw("docs/a.md", "hSame")])
    store = MagicMock()
    meta = MagicMock()
    meta.all_sync_state.return_value = {"docs/a.md": {"file_hash": "hSame", "mtime": 1.0}}

    rep = SyncService(pipeline, store, meta).sync()

    pipeline.process_file.assert_not_called()
    store.delete_by_file_hash.assert_not_called()
    assert rep.total == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_sync_service.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'fin_rag.sync'`）

- [ ] **Step 3: 重构 `src/fin_rag/pipeline/pipeline.py`（抽出 `process_file`）**

把 `IndexPipeline.run` 的单文件处理逻辑抽成方法（`run` 改为循环调用它），新增：

```python
    def process_file(self, datasource, raw, cfg) -> int:
        """处理单个文件：解析→切分→打标→向量化→入库，返回入库切块数。"""
        if not self.router.supports(raw.rel_path):
            return 0
        content = datasource.read(raw)
        doc = self.router.parse(raw, content)
        nodes = self.chunker.chunk(doc, file_hash=raw.file_hash)
        nodes = [self.extractor.enrich(_with_mtime(n, raw.mtime), cfg) for n in nodes]
        if not nodes:
            return 0
        vectors = self.embedder.embed([n.text for n in nodes])
        self.store.upsert_hybrid(nodes, vectors)
        return len(nodes)
```

并把 `run` 内的循环体替换为调用它：

```python
    def run(self) -> int:
        self.store.ensure_hybrid_collection()
        total = 0
        for cfg in self.sources:
            ds = self.datasource_factory(cfg)
            for raw in ds.list_files():
                total += self.process_file(ds, raw, cfg)
        return total
```

（`_with_mtime` 保留在模块底部不变。）

- [ ] **Step 4: 在 `src/fin_rag/store/hybrid_store.py` 加 `delete_by_file_hash`**

在 `HybridVectorStore` 内追加（按 project + file_hash 删除该文件的所有旧切块）：

```python
    def delete_by_file_hash(self, project: str, file_hash: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="project", match=models.MatchValue(value=project)),
                    models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash)),
                ]
            )
        )
        self._client.delete(
            collection_name=self.settings.qdrant_collection, points_selector=selector
        )
```

- [ ] **Step 5: 写 `src/fin_rag/sync/__init__.py`（空）**

```python
```

- [ ] **Step 6: 写 `src/fin_rag/sync/sync_service.py`**

```python
"""增量同步：hash+mtime 幂等。新增→索引；变更→删旧 hash 切块再索引；删除→清理。"""

from dataclasses import dataclass

from fin_rag.pipeline.pipeline import IndexPipeline
from fin_rag.store.hybrid_store import HybridVectorStore
from fin_rag.store.metadata_store import MetadataStore


@dataclass(frozen=True)
class SyncReport:
    added: int
    updated: int
    deleted: int

    @property
    def total(self) -> int:
        return self.added + self.updated + self.deleted


class SyncService:
    def __init__(
        self,
        pipeline: IndexPipeline,
        store: HybridVectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        self.pipeline = pipeline
        self.store = store
        self.meta = metadata_store

    def sync(self) -> SyncReport:
        added = updated = deleted = 0
        for cfg in self.pipeline.sources:
            ds = self.pipeline.datasource_factory(cfg)
            current = {f.rel_path: f for f in ds.list_files()}
            known = self.meta.all_sync_state(cfg.name)

            for rel, raw in current.items():
                prev = known.get(rel)
                if prev is None:
                    self.pipeline.process_file(ds, raw, cfg)
                    self.meta.set_sync_state(cfg.name, rel, raw.file_hash, raw.mtime)
                    added += 1
                elif prev["file_hash"] != raw.file_hash:
                    self.store.delete_by_file_hash(cfg.name, prev["file_hash"])
                    self.pipeline.process_file(ds, raw, cfg)
                    self.meta.set_sync_state(cfg.name, rel, raw.file_hash, raw.mtime)
                    updated += 1

            for rel, prev in known.items():
                if rel not in current:
                    self.store.delete_by_file_hash(cfg.name, prev["file_hash"])
                    self.meta.delete_sync_state(cfg.name, rel)
                    deleted += 1

        return SyncReport(added=added, updated=updated, deleted=deleted)
```

- [ ] **Step 7: 运行测试确认通过**

Run: `pytest tests/test_sync_service.py tests/test_pipeline.py -v`
Expected: PASS（sync 4 + pipeline 原有；确认 run 重构未破坏 M2 Task8）

- [ ] **Step 8: 改 `src/fin_rag/cli.py` 加 `sync` 子命令**

import 区加（`MetadataStore`/`HybridVectorStore`/`_build_pipeline` 已在；`SyncService`/`IndexPipeline` 视已有 import 补）：

```python
from fin_rag.pipeline.pipeline import IndexPipeline
from fin_rag.sync.sync_service import SyncService
```

在 `_build_web_app()` 之后追加工厂：

```python
def _build_sync_service():
    settings = load_settings(CONFIG_DIR / "settings.yaml")
    pipeline = _build_pipeline()          # 复用 index 的 IndexPipeline（已 hybrid）
    store = HybridVectorStore(settings)
    meta = MetadataStore()
    return SyncService(pipeline=pipeline, store=store, metadata_store=meta)
```

在 `main()` subparser 区（`serve` 之后）加：

```python
    sub.add_parser("sync", help="增量同步（hash+mtime 幂等）")
```

在 `if args.cmd == "serve":` 分支之后加：

```python
    if args.cmd == "sync":
        try:
            rep = _build_sync_service().sync()
            log.info("增量同步完成：新增 %d · 更新 %d · 删除 %d", rep.added, rep.updated, rep.deleted)
            return 0
        except Exception as exc:
            log.error("同步失败：%s", exc)
            return 1
```

> 注：`_build_pipeline`（M0-M1）返回 `IndexPipeline`；M2 Task8 已将其 store 换为 `HybridVectorStore`，故 `SyncService` 直接复用，写入与 index 一致。

- [ ] **Step 9: Commit**

```bash
git add src/fin_rag/pipeline/pipeline.py src/fin_rag/store/hybrid_store.py src/fin_rag/sync/ src/fin_rag/cli.py tests/test_sync_service.py
git commit -m "feat: 增量同步 SyncService（hash+mtime 幂等，新增/变更/删除）+ CLI sync"
```

---

## Task 4: Docker 部署（app + qdrant）

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Modify: `docker-compose.yml`（加 `app` 服务）
- Modify: `src/fin_rag/config.py`（`QDRANT_URL` 环境变量优先）

- [ ] **Step 1: 写 `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 依赖先装（利用层缓存）
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY eval ./eval

RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["fin-rag", "serve", "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] **Step 2: 写 `.dockerignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.git/
data/
docs/
tests/
*.egg-info/
.env
.pytest_cache/
.coverage
htmlcov/
```

- [ ] **Step 3: 改 `docker-compose.yml`（在 M0-M1 的 qdrant 基础上加 app）**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage
    environment:
      QDRANT__LOG_LEVEL: INFO

  app:
    build: .
    depends_on:
      - qdrant
    ports:
      - "7860:7860"
    environment:
      ZHIPU_API_KEY: ${ZHIPU_API_KEY}
      DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY}
      QDRANT_URL: http://qdrant:6333
    volumes:
      - ./data:/app/data         # SQLite 元数据 + 持久化
      - ./config:/app/config     # 配置可热改
```

- [ ] **Step 4: 改 `src/fin_rag/config.py` 让 `QDRANT_URL` 环境变量优先（容器内指向 qdrant 服务）**

在 `load_settings` 内，`qdrant_url` 的取值改为环境变量优先：

```python
import os
...
        qdrant_url=os.environ.get("QDRANT_URL", qdrant.get("url", "http://localhost:6333")),
```

（`Settings` 是 `BaseSettings`，环境变量本就优先；但 `load_settings` 显式传参会覆盖 env，此处让 env 重新优先，确保 compose 的 `QDRANT_URL=http://qdrant:6333` 生效。）

- [ ] **Step 5: 部署验证（一键起 + 建库 + 访问）**

```bash
cd /Users/wangfei/yulore/fin-rag
cp .env.example .env  # 填真实 ZHIPU_API_KEY / DEEPSEEK_API_KEY
docker compose build
docker compose up -d
# 容器内首次建库（API key 已通过 env 注入）
docker compose exec app fin-rag index
# 评测（可选）
docker compose exec app python eval/run_eval.py
```
浏览器开 `http://<服务器IP>:7860`：
- 多轮问答、溯源、防幻觉拒答、反馈均正常。
- `docker compose down` 后 `./data` 保留（qdrant 向量 + SQLite 元数据），再 `up` 不丢数据。

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml src/fin_rag/config.py
git commit -m "feat: Docker 部署（app+qdrant 一键起，env 注入密钥，数据卷持久化）"
```

---

## 完成标准（M5）

- [ ] Task1-3 测试通过，`pytest` 全量覆盖率 ≥ 80%（Task4 Docker 仅手动验证）
- [ ] `python eval/run_eval.py` 对 Golden Set 输出四项指标 + `last_report.json`
- [ ] `fin-rag sync` 增量正确：改一个文件只更新它、删一个文件只清理它、未变不动
- [ ] `docker compose up` 一键起 app + qdrant，浏览器可用，`down` 后数据不丢

---

## 全部五份计划总览

| 阶段 | 文件 | 产出 | 可验证命令 |
|------|------|------|-----------|
| M0-M1 | `…-m0-m1-index-pipeline.md` | 索引管道 | `fin-rag index` |
| M2 | `…-m2-retrieval.md`（含 Task8 修正） | 混合检索 | `fin-rag search` |
| M3 | `…-m3-qa-orchestration.md` | 防幻觉问答 | `fin-rag ask` |
| M4 | `…-m4-frontend.md` | Gradio 前端 | `fin-rag serve` |
| M5 | `…-m5-eval-sync-deploy.md` | 评测/同步/部署 | `run_eval.py` / `fin-rag sync` / `docker compose up` |

至此 spec 的 M0-M5 全部落地为可执行 TDD 计划。**执行建议**：按 M0-M1 → M2 → M3 → M4 → M5 顺序，每阶段做完跑通对应命令再进下一阶段（每份计划都在末尾给了端到端验证步骤）。
