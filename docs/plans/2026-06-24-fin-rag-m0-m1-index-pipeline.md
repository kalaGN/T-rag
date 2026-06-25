# fin-rag M0-M1 实现计划：脚手架 + 配置 + 数据管道 + 索引

> **已废弃（2026-06-25）**：本计划基于自建索引管道，与当前 LlamaIndex 技术方案不兼容，请勿执行。后续按 `docs/design/2026-06-24-fin-rag-design.md` 重新拆分实施计划。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 fin-rag 地基——扫描 fin-online/opdata 两个项目（应用 bxf/testcase/review 排除规则），解析 md/mdc/csv/xlsx，结构化切分，打 domain/doc_type/产品号/渠道 元数据，向量化后写入 Qdrant，产出可查询验证的索引。

**Architecture:** 分层管道 `Collector → FormatAdapter → Chunker → MetadataExtractor → Embedder → VectorStore`，全程不可变数据模型，配置驱动多数据源。本阶段不含检索/问答（属 M2-M3 计划）。

**Tech Stack:** Python 3.11+、Pydantic/pydantic-settings（配置与不可变模型）、PyYAML、markdown-it-py、python-frontmatter、openpyxl、qdrant-client、openai SDK（DeepSeek LLM + 智谱 Embedding-3 embedding，均为 OpenAI 兼容）、pytest（≥80%）、Docker（Qdrant）。

**关键接口约定（跨任务必须保持一致）：**
- `SourceConfig(name, root, include, default_domain, exclude_patterns)`
- `RawFile(source, rel_path, abs_path, mtime, file_hash)`
- `Document(source, rel_path, text, sections, frontmatter)`
- `Section(level, title, text, heading_path)` — 切分单元
- `Node(node_id, source, rel_path, text, chunk_index, project, domain, source_type, doc_type, product_code, channel, heading_path, file_hash, mtime, parent_node_id)`
- `node_id` / `chunk_id = f"{project}:{file_hash}:{chunk_index}"`（幂等）

**前置准备（人工，执行前完成）：**
1. 获取 DeepSeek API Key（`DEEPSEEK_API_KEY`）
2. 获取智谱 API Key（`ZHIPU_API_KEY`，格式为 `id.secret`）
3. 本机已安装 Docker（用于起 Qdrant）

---

## Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `src/fin_rag/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: 初始化 git 仓库与目录结构**

Run:
```bash
cd /Users/wangfei/yulore/fin-rag
git init
mkdir -p src/fin_rag tests config
```

- [ ] **Step 2: 写冒烟测试 `tests/test_smoke.py`**

```python
def test_package_importable():
    import fin_rag

    assert fin_rag.__version__
```

- [ ] **Step 3: 写 `src/fin_rag/__init__.py`**

```python
"""fin-rag: 运营/金融知识库智能问答助手。"""

__version__ = "0.1.0"
```

- [ ] **Step 4: 写 `tests/__init__.py`（空文件）**

```python
```

- [ ] **Step 5: 写 `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fin-rag"
version = "0.1.0"
description = "运营/金融知识库智能问答助手"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "markdown-it-py>=3.0",
    "python-frontmatter>=1.1",
    "openpyxl>=3.1",
    "qdrant-client>=1.11",
    "openai>=1.30",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[project.scripts]
fin-rag = "fin_rag.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/fin_rag"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "--cov=fin_rag --cov-report=term-missing"
```

- [ ] **Step 6: 写 `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.env
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
data/qdrant_storage/
```

- [ ] **Step 7: 写 `.env.example`**

```dotenv
# LLM（问答，M3 使用，M0-M1 不强制）
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Embedding（智谱 Embedding-3，OpenAI 兼容；key 格式 id.secret）
ZHIPU_API_KEY=xxxxxxxx.xxxxxxxx
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3

# 向量库
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=fin_rag
EMBEDDING_DIM=1024
```

- [ ] **Step 8: 写 `docker-compose.yml`（Qdrant）**

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
```

- [ ] **Step 9: 安装依赖并验证冒烟测试通过**

Run:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```
Expected: PASS（1 passed）

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: 项目脚手架（pyproject、依赖、Qdrant compose、冒烟测试）"
```

---

## Task 2: 配置系统

**Files:**
- Create: `src/fin_rag/config.py`
- Create: `config/settings.yaml`
- Create: `config/sources.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
from pathlib import Path

import pytest

from fin_rag.config import Settings, SourceConfig, load_settings, load_sources

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_load_settings_reads_yaml_and_env_defaults(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    settings = load_settings(CONFIG_DIR / "settings.yaml")

    assert isinstance(settings, Settings)
    assert settings.embedding_model == "embedding-3"
    assert settings.embedding_dim == 1024
    assert settings.qdrant_collection == "fin_rag"


def test_load_sources_parses_two_projects():
    sources = load_sources(CONFIG_DIR / "sources.yaml")

    assert len(sources) == 2
    names = {s.name for s in sources}
    assert names == {"fin-online", "opdata"}

    fin = next(s for s in sources if s.name == "fin-online")
    assert "docs/" in fin.include
    assert "*bxf*" in fin.exclude_patterns
    assert "docs/testcase/" in fin.exclude_patterns
    assert "docs/review/" in fin.exclude_patterns
    assert fin.default_domain == "fin-online"


def test_source_config_is_immutable():
    cfg = SourceConfig(
        name="x", root="/tmp", include=["docs/"], default_domain="x"
    )
    with pytest.raises(Exception):
        cfg.name = "y"  # frozen 不可变
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'fin_rag.config'`）

- [ ] **Step 3: 写 `config/settings.yaml`**

```yaml
embedding:
  model: embedding-3
  dim: 1024
  batch_size: 32

qdrant:
  url: http://localhost:6333
  collection: fin_rag

deepseek:
  base_url: https://api.deepseek.com
  model: deepseek-chat
```

- [ ] **Step 4: 写 `config/sources.yaml`**

```yaml
sources:
  - name: fin-online
    root: /Users/wangfei/yulore/fin-online.dianhua.cn
    include: [docs/, .cursor/]
    exclude_patterns:
      - "*bxf*"
      - "docs/testcase/"
      - "docs/review/"
      - ".DS_Store"
      - "target/"
      - ".git/"
    default_domain: fin-online
  - name: opdata
    root: /Users/wangfei/yulore/opdata.dianhua.cn
    include: [docs/, .cursor/]
    exclude_patterns:
      - ".DS_Store"
      - "target/"
      - ".git/"
    default_domain: opdata
```

- [ ] **Step 5: 写 `src/fin_rag/config.py`**

```python
"""配置加载：settings.yaml + sources.yaml + 环境变量，全部不可变。"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class SourceConfig:
    """单个数据源（项目）的采集配置，不可变。"""

    name: str
    root: str
    include: list[str]
    default_domain: str
    exclude_patterns: list[str] = field(default_factory=list)


class Settings(BaseSettings):
    """运行时配置，环境变量优先（API key 等敏感信息不写死）。"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embedding（智谱 Embedding-3，OpenAI 兼容）
    zhipu_api_key: str
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    embedding_model: str = "embedding-3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32

    # LLM（DeepSeek，M3 使用）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # 向量库
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "fin_rag"


def load_settings(settings_yaml: Path) -> Settings:
    """从 settings.yaml 读默认值，环境变量覆盖，返回 Settings。"""
    raw = yaml.safe_load(settings_yaml.read_text(encoding="utf-8")) or {}
    embedding = raw.get("embedding", {})
    qdrant = raw.get("qdrant", {})
    deepseek = raw.get("deepseek", {})
    return Settings(
        embedding_model=embedding.get("model", "embedding-3"),
        embedding_dim=embedding.get("dim", 1024),
        embedding_batch_size=embedding.get("batch_size", 32),
        qdrant_url=qdrant.get("url", "http://localhost:6333"),
        qdrant_collection=qdrant.get("collection", "fin_rag"),
        deepseek_base_url=deepseek.get("base_url", "https://api.deepseek.com"),
        deepseek_model=deepseek.get("model", "deepseek-chat"),
    )


def load_sources(sources_yaml: Path) -> list[SourceConfig]:
    """从 sources.yaml 加载数据源清单，返回不可变 SourceConfig 列表。"""
    raw = yaml.safe_load(sources_yaml.read_text(encoding="utf-8")) or {}
    items = raw.get("sources", [])
    return [
        SourceConfig(
            name=item["name"],
            root=item["root"],
            include=list(item.get("include", [])),
            default_domain=item.get("default_domain", item["name"]),
            exclude_patterns=list(item.get("exclude_patterns", [])),
        )
        for item in items
    ]
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS（3 passed）

- [ ] **Step 7: Commit**

```bash
git add src/fin_rag/config.py config/ tests/test_config.py
git commit -m "feat: 配置系统（settings/sources yaml + Pydantic 不可变配置）"
```

---

## Task 3: 不可变数据模型

**Files:**
- Create: `src/fin_rag/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 写失败测试 `tests/test_models.py`**

```python
import pytest

from fin_rag.models import Document, Node, RawFile, Section


def test_raw_file_is_immutable():
    rf = RawFile(
        source="fin-online",
        rel_path="docs/a.md",
        abs_path="/tmp/docs/a.md",
        mtime=1.0,
        file_hash="abc",
    )
    with pytest.raises(Exception):
        rf.rel_path = "b.md"


def test_section_carries_heading_path():
    sec = Section(level=2, title="鉴权", text="内容", heading_path="接口/鉴权")
    assert sec.heading_path == "接口/鉴权"


def test_node_factory_builds_id_from_project_hash_index():
    node = Node(
        source="fin-online",
        rel_path="docs/a.md",
        text="t",
        chunk_index=3,
        project="fin-online",
        domain="fin-online",
        source_type="doc",
        doc_type="tech_plan",
        product_code=None,
        channel=None,
        heading_path="",
        file_hash="abc123",
        mtime=1.0,
        parent_node_id=None,
    )
    assert node.node_id == "fin-online:abc123:3"


def test_node_is_immutable():
    node = Node(
        source="fin-online", rel_path="d", text="t", chunk_index=0,
        project="fin-online", domain="fin-online", source_type="doc",
        doc_type="tech_plan", product_code=None, channel=None,
        heading_path="", file_hash="h", mtime=0.0, parent_node_id=None,
    )
    with pytest.raises(Exception):
        node.domain = "opdata"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'fin_rag.models'`）

- [ ] **Step 3: 写 `src/fin_rag/models.py`**

```python
"""全链路不可变数据模型（frozen dataclass）。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RawFile:
    """采集器产出的原始文件清单项。"""

    source: str
    rel_path: str
    abs_path: str
    mtime: float
    file_hash: str


@dataclass(frozen=True)
class Section:
    """按标题树切分后的章节单元。"""

    level: int
    title: str
    text: str
    heading_path: str  # 形如 "接口设计/鉴权"


@dataclass(frozen=True)
class Document:
    """格式适配器解析后的统一中间结构。"""

    source: str
    rel_path: str
    text: str
    sections: list[Section]
    frontmatter: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Node:
    """切块 + 元数据，索引入库的基本单元，不可变。"""

    source: str
    rel_path: str
    text: str
    chunk_index: int
    project: str
    domain: str
    source_type: str
    doc_type: str
    product_code: Optional[str]
    channel: Optional[str]
    heading_path: str
    file_hash: str
    mtime: float
    parent_node_id: Optional[str] = None

    @property
    def node_id(self) -> str:
        """幂等 id：project:file_hash:chunk_index。"""
        return f"{self.project}:{self.file_hash}:{self.chunk_index}"


def make_node(**kwargs) -> Node:
    """工厂，补默认值，便于链路构造。"""
    defaults = dict(
        source="", rel_path="", text="", chunk_index=0, project="",
        domain="", source_type="doc", doc_type="tech_plan",
        product_code=None, channel=None, heading_path="",
        file_hash="", mtime=0.0, parent_node_id=None,
    )
    defaults.update(kwargs)
    return Node(**defaults)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_models.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/models.py tests/test_models.py
git commit -m "feat: 不可变数据模型（RawFile/Section/Document/Node）"
```

---

## Task 4: DataSource 抽象 + FilesystemDataSource（扫描 + 排除 + hash）

**Files:**
- Create: `src/fin_rag/pipeline/__init__.py`
- Create: `src/fin_rag/pipeline/datasource.py`
- Test: `tests/test_datasource.py`
- Test fixture: `tests/fixtures/sample_project/docs/keep.md` 等（见 Step 1）

- [ ] **Step 1: 准备测试 fixture 目录树**

Run:
```bash
cd /Users/wangfei/yulore/fin-rag
mkdir -p tests/fixtures/sample_project/docs
mkdir -p tests/fixtures/sample_project/docs/plan/bxf
mkdir -p tests/fixtures/sample_project/docs/testcase
mkdir -p tests/fixtures/sample_project/.cursor/rules
printf '# keep\n正文\n' > tests/fixtures/sample_project/docs/keep.md
printf '# bxf\nbxf内容\n' > tests/fixtures/sample_project/docs/plan/bxf/20260101-bxf方案.md
printf '# 用例\n' > tests/fixtures/sample_project/docs/testcase/用例.md
printf '# rule\n' > tests/fixtures/sample_project/.cursor/rules/core.mdc
printf 'binary' > tests/fixtures/sample_project/docs/.DS_Store
```

- [ ] **Step 2: 写失败测试 `tests/test_datasource.py`**

```python
from pathlib import Path

from fin_rag.config import SourceConfig
from fin_rag.pipeline.datasource import FilesystemDataSource

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_project"


def _cfg():
    return SourceConfig(
        name="sample",
        root=str(FIXTURE),
        include=["docs/", ".cursor/"],
        exclude_patterns=["*bxf*", "docs/testcase/", ".DS_Store"],
        default_domain="sample",
    )


def test_list_files_applies_excludes():
    ds = FilesystemDataSource(_cfg())
    files = ds.list_files()
    rels = sorted(f.rel_path for f in files)

    assert "docs/keep.md" in rels
    assert ".cursor/rules/core.mdc" in rels
    # 排除：bxf、testcase、.DS_Store
    assert not any("bxf" in r.lower() for r in rels)
    assert not any("testcase" in r for r in rels)
    assert not any(".DS_Store" in r for r in rels)


def test_file_hash_is_deterministic():
    ds = FilesystemDataSource(_cfg())
    files = ds.list_files()
    keep = next(f for f in files if f.rel_path == "docs/keep.md")
    again = next(f for f in FilesystemDataSource(_cfg()).list_files()
                 if f.rel_path == "docs/keep.md")
    assert keep.file_hash == again.file_hash
    assert len(keep.file_hash) > 0


def test_read_returns_bytes():
    ds = FilesystemDataSource(_cfg())
    files = ds.list_files()
    keep = next(f for f in files if f.rel_path == "docs/keep.md")
    content = ds.read(keep)
    assert b"\xe6\xad\xa3\xe6\x96\x87" in content  # "正文" utf-8
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_datasource.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 4: 写 `src/fin_rag/pipeline/__init__.py`（空）**

```python
```

- [ ] **Step 5: 写 `src/fin_rag/pipeline/datasource.py`**

```python
"""数据源抽象与文件系统实现：扫描、排除规则、hash/mtime。"""

import hashlib
from fnmatch import fnmatch
from pathlib import Path
from typing import Protocol

from fin_rag.config import SourceConfig
from fin_rag.models import RawFile

# 不索引的扩展名/文件（兜底，与配置排除规则叠加）
_SKIP_EXACT = {".DS_Store", "Thumbs.db"}
_SKIP_DIRS = {".git", "target", "node_modules", "__pycache__"}


class DataSource(Protocol):
    """数据源抽象。首版实现 FilesystemDataSource；二期 MySQLDataSource。"""

    name: str

    def list_files(self) -> list[RawFile]: ...

    def read(self, file: RawFile) -> bytes: ...


def _is_excluded(rel_path: str, patterns: list[str]) -> bool:
    """支持 glob（*bxf*）与路径前缀（docs/testcase/）两种排除。"""
    normalized = rel_path.replace("\\", "/")
    for pat in patterns:
        pat_norm = pat.replace("\\", "/")
        if pat_norm.endswith("/"):
            if normalized.startswith(pat_norm) or f"/{pat_norm}" in "/" + normalized:
                return True
        elif fnmatch(normalized, f"*{pat_norm}*") or fnmatch(normalized, pat_norm):
            return True
    return False


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FilesystemDataSource:
    """文件系统数据源：扫描 include 目录，应用 exclude_patterns。"""

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.root = Path(cfg.root)

    def list_files(self) -> list[RawFile]:
        results: list[RawFile] = []
        for include_dir in self.cfg.include:
            base = self.root / include_dir.rstrip("/")
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                if path.name in _SKIP_EXACT or any(
                    part in _SKIP_DIRS for part in path.parts
                ):
                    continue
                rel = self._rel_under_root(path)
                if _is_excluded(rel, self.cfg.exclude_patterns):
                    continue
                content = path.read_bytes()
                results.append(
                    RawFile(
                        source=self.name,
                        rel_path=rel,
                        abs_path=str(path),
                        mtime=path.stat().st_mtime,
                        file_hash=_sha256(content),
                    )
                )
        return results

    def read(self, file: RawFile) -> bytes:
        return Path(file.abs_path).read_bytes()

    def _rel_under_root(self, path: Path) -> str:
        return str(path.relative_to(self.root)).replace("\\", "/")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_datasource.py -v`
Expected: PASS（3 passed）

- [ ] **Step 7: Commit**

```bash
git add src/fin_rag/pipeline tests/fixtures tests/test_datasource.py
git commit -m "feat: DataSource 抽象 + FilesystemDataSource（扫描/bxf·testcase排除/hash）"
```

---

## Task 5: Markdown 适配器（标题树解析）

**Files:**
- Create: `src/fin_rag/pipeline/adapters/__init__.py`
- Create: `src/fin_rag/pipeline/adapters/base.py`
- Create: `src/fin_rag/pipeline/adapters/markdown.py`
- Test: `tests/test_adapter_markdown.py`

- [ ] **Step 1: 写失败测试 `tests/test_adapter_markdown.py`**

```python
from fin_rag.models import RawFile
from fin_rag.pipeline.adapters.markdown import MarkdownAdapter


def _raw(text: str) -> RawFile:
    return RawFile(
        source="s", rel_path="docs/a.md", abs_path="/a.md",
        mtime=0.0, file_hash="h",
    )


def test_markdown_adapter_supports_md_only():
    ad = MarkdownAdapter()
    assert ad.supports("docs/a.md")
    assert not ad.supports("a.txt")


def test_parse_builds_sections_by_headings():
    text = "# 总览\n简介\n## 接口设计\n细节\n## 鉴权\nkey\n"
    doc = MarkdownAdapter().parse(_raw(text), text.encode("utf-8"))

    assert doc.source == "s"
    assert doc.rel_path == "docs/a.md"
    titles = [s.title for s in doc.sections]
    assert titles == ["总览", "接口设计", "鉴权"]
    # heading_path 含父级
    auth = next(s for s in doc.sections if s.title == "鉴权")
    assert "总览" in auth.heading_path


def test_parse_empty_doc_yields_no_sections():
    doc = MarkdownAdapter().parse(_raw(""), b"")
    assert doc.sections == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_adapter_markdown.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/pipeline/adapters/__init__.py`（空）**

```python
```

- [ ] **Step 4: 写 `src/fin_rag/pipeline/adapters/base.py`**

```python
"""格式适配器抽象。"""

from typing import Protocol

from fin_rag.models import Document, RawFile


class FormatAdapter(Protocol):
    def supports(self, rel_path: str) -> bool: ...

    def parse(self, file: RawFile, content: bytes) -> Document: ...
```

- [ ] **Step 5: 写 `src/fin_rag/pipeline/adapters/markdown.py`**

```python
"""Markdown 适配器：按标题树解析为 Section 列表。"""

from markdown_it import MarkdownIt

from fin_rag.models import Document, RawFile, Section


class MarkdownAdapter:
    def supports(self, rel_path: str) -> bool:
        return rel_path.lower().endswith(".md")

    def parse(self, file: RawFile, content: bytes) -> Document:
        text = content.decode("utf-8", errors="replace")
        sections = self._split_by_headings(text)
        return Document(
            source=file.source,
            rel_path=file.rel_path,
            text=text,
            sections=sections,
        )

    def _split_by_headings(self, text: str) -> list[Section]:
        md = MarkdownIt()
        tokens = md.parse(text)
        path_stack: list[tuple[int, str]] = []  # (level, title)
        results: list[Section] = []
        current_title: str | None = None
        current_level: int = 0
        current_lines: list[str] = []

        def _flush() -> None:
            nonlocal current_title, current_level, current_lines
            if current_title is not None:
                heading_path = "/".join(t for _, t in path_stack)
                results.append(
                    Section(
                        level=current_level,
                        title=current_title,
                        text="".join(current_lines).strip(),
                        heading_path=heading_path,
                    )
                )
            current_title = None
            current_lines = []

        for token in tokens:
            if token.type == "heading_open":
                _flush()
                current_level = int(token.tag[1:])
            elif token.type == "inline" and current_title is None and current_level:
                current_title = token.content
                while path_stack and path_stack[-1][0] >= current_level:
                    path_stack.pop()
                path_stack.append((current_level, current_title))
            elif token.type == "inline":
                current_lines.append(token.content)
            elif token.type == "paragraph_open":
                current_lines.append("\n")
        _flush()
        return results
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_adapter_markdown.py -v`
Expected: PASS（3 passed）

- [ ] **Step 7: Commit**

```bash
git add src/fin_rag/pipeline/adapters tests/test_adapter_markdown.py
git commit -m "feat: Markdown 适配器（标题树解析为 Section）"
```

---

## Task 6: MDC 适配器（剥离 YAML frontmatter）

**Files:**
- Create: `src/fin_rag/pipeline/adapters/mdc.py`
- Test: `tests/test_adapter_mdc.py`

- [ ] **Step 1: 写失败测试 `tests/test_adapter_mdc.py`**

```python
from fin_rag.pipeline.adapters.mdc import MdcAdapter

MDC = """---
description: 防幻觉规则
globs: docs/**/*.md
alwaysApply: true
---

# 防幻觉

只基于上下文作答。
"""


def test_mdc_adapter_supports_mdc_only():
    ad = MdcAdapter()
    assert ad.supports("rules/core.mdc")
    assert not ad.supports("core.md")


def test_parse_strips_frontmatter_into_metadata():
    from fin_rag.models import RawFile
    raw = RawFile("s", "rules/core.mdc", "/c", 0.0, "h")
    doc = MdcAdapter().parse(raw, MDC.encode("utf-8"))

    # 正文不含 frontmatter
    assert "---" not in doc.text
    assert "防幻觉" in doc.text
    # frontmatter 抽取
    assert doc.frontmatter.get("alwaysApply") is True
    assert doc.frontmatter.get("description") == "防幻觉规则"
    assert doc.frontmatter.get("globs") == "docs/**/*.md"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_adapter_mdc.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/pipeline/adapters/mdc.py`**

```python
"""MDC（Cursor 规则）适配器：剥离 YAML frontmatter 入 metadata。"""

import frontmatter
from markdown_it import MarkdownIt

from fin_rag.models import Document, RawFile, Section


class MdcAdapter:
    def supports(self, rel_path: str) -> bool:
        return rel_path.lower().endswith(".mdc")

    def parse(self, file: RawFile, content: bytes) -> Document:
        text = content.decode("utf-8", errors="replace")
        post = frontmatter.loads(text)
        body = post.content
        sections = self._split_by_headings(body)
        return Document(
            source=file.source,
            rel_path=file.rel_path,
            text=body,
            sections=sections,
            frontmatter=dict(post.metadata),
        )

    def _split_by_headings(self, text: str) -> list[Section]:
        md = MarkdownIt()
        tokens = md.parse(text)
        path_stack: list[tuple[int, str]] = []
        results: list[Section] = []
        current_title: str | None = None
        current_level: int = 0
        current_lines: list[str] = []

        def _flush() -> None:
            nonlocal current_title, current_level, current_lines
            if current_title is not None:
                heading_path = "/".join(t for _, t in path_stack)
                results.append(
                    Section(
                        level=current_level,
                        title=current_title,
                        text="".join(current_lines).strip(),
                        heading_path=heading_path,
                    )
                )
            current_title = None
            current_lines = []

        for token in tokens:
            if token.type == "heading_open":
                _flush()
                current_level = int(token.tag[1:])
            elif token.type == "inline" and current_title is None and current_level:
                current_title = token.content
                while path_stack and path_stack[-1][0] >= current_level:
                    path_stack.pop()
                path_stack.append((current_level, current_title))
            elif token.type == "inline":
                current_lines.append(token.content)
            elif token.type == "paragraph_open":
                current_lines.append("\n")
        _flush()
        return results
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_adapter_mdc.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/pipeline/adapters/mdc.py tests/test_adapter_mdc.py
git commit -m "feat: MDC 适配器（剥离 YAML frontmatter 入 metadata）"
```

---

## Task 7: CSV/XLSX 适配器（小表行组、大表摘要）

**Files:**
- Create: `src/fin_rag/pipeline/adapters/tabular.py`
- Test: `tests/test_adapter_tabular.py`

- [ ] **Step 1: 写失败测试 `tests/test_adapter_tabular.py`**

```python
import io

from openpyxl import Workbook

from fin_rag.models import RawFile
from fin_rag.pipeline.adapters.tabular import CsvAdapter, XlsxAdapter, LARGE_TABLE_THRESHOLD


def _raw(name: str) -> RawFile:
    return RawFile("s", name, "/" + name, 0.0, "h")


def test_csv_small_table_rows_grouped():
    csv_text = "产品号,渠道\n300007,cucc\n300006,cmcc\n"
    doc = CsvAdapter().parse(_raw("a.csv"), csv_text.encode("utf-8"))
    assert any("300007" in s.text and "cucc" in s.text for s in doc.sections)
    assert any("300006" in s.text for s in doc.sections)


def test_csv_large_table_only_summary():
    header = "c1,c2\n"
    big = (header + "1,2\n") * (LARGE_TABLE_THRESHOLD + 5)
    doc = CsvAdapter().parse(_raw("big.csv"), big.encode("utf-8"))
    # 大表：不逐行入全文，仅摘要（section 数量很少）
    assert len(doc.sections) <= 2
    assert "行" in doc.sections[0].text or "摘要" in doc.sections[0].text


def test_xlsx_one_section_per_sheet():
    wb = Workbook()
    ws = wb.active
    ws.title = "产品"
    ws.append(["产品号", "渠道"])
    ws.append(["300007", "cucc"])
    buf = io.BytesIO()
    wb.save(buf)
    doc = XlsxAdapter().parse(_raw("a.xlsx"), buf.getvalue())
    titles = [s.title for s in doc.sections]
    assert "产品" in titles
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_adapter_tabular.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/pipeline/adapters/tabular.py`**

```python
"""CSV/XLSX 适配器：小表按行成组，大表仅建摘要。"""

import csv
import io

from openpyxl import load_workbook

from fin_rag.models import Document, RawFile, Section

LARGE_TABLE_THRESHOLD = 500  # 行数阈值，超过则仅摘要


class CsvAdapter:
    def supports(self, rel_path: str) -> bool:
        return rel_path.lower().endswith(".csv")

    def parse(self, file: RawFile, content: bytes) -> Document:
        text = content.decode("utf-8-sig", errors="replace")
        reader = list(csv.reader(io.StringIO(text)))
        sections = self._build_sections(reader)
        return Document(
            source=file.source, rel_path=file.rel_path, text=text, sections=sections
        )

    def _build_sections(self, rows: list[list[str]]) -> list[Section]:
        if not rows:
            return []
        header = ",".join(rows[0])
        data = rows[1:]
        if len(data) > LARGE_TABLE_THRESHOLD:
            summary = (
                f"摘要：表头[{header}]，共 {len(data)} 行（超过阈值，仅建摘要不入全文）"
            )
            return [Section(level=1, title="表摘要", text=summary, heading_path="表摘要")]
        sections: list[Section] = []
        for i, row in enumerate(data, start=1):
            row_text = f"表头: {header} | 第{i}行: {','.join(row)}"
            sections.append(
                Section(level=2, title=f"行{i}", text=row_text, heading_path=f"表/行{i}")
            )
        return sections


class XlsxAdapter:
    def supports(self, rel_path: str) -> bool:
        return rel_path.lower().endswith(".xlsx")

    def parse(self, file: RawFile, content: bytes) -> Document:
        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        sections: list[Section] = []
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            if not rows:
                continue
            header = ",".join(str(c) for c in rows[0])
            data = rows[1:]
            if len(data) > LARGE_TABLE_THRESHOLD:
                text = (
                    f"sheet[{ws.title}] 表头[{header}] 共{len(data)}行"
                    f"（超过阈值，仅建摘要）"
                )
            else:
                text = f"sheet[{ws.title}] 表头: {header}\n" + "\n".join(
                    ",".join(str(c) for c in r) for r in data
                )
            sections.append(
                Section(level=1, title=ws.title, text=text, heading_path=f"sheet/{ws.title}")
            )
        wb.close()
        return Document(
            source=file.source,
            rel_path=file.rel_path,
            text="",
            sections=sections,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_adapter_tabular.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/pipeline/adapters/tabular.py tests/test_adapter_tabular.py
git commit -m "feat: CSV/XLSX 适配器（小表行组、大表仅摘要）"
```

---

## Task 8: 适配器路由（FormatRouter 按扩展名分发）

**Files:**
- Create: `src/fin_rag/pipeline/adapters/router.py`
- Test: `tests/test_adapter_router.py`

- [ ] **Step 1: 写失败测试 `tests/test_adapter_router.py`**

```python
import pytest

from fin_rag.models import Document, RawFile
from fin_rag.pipeline.adapters.router import FormatRouter


class _FakeMd:
    def supports(self, p: str) -> bool:
        return p.endswith(".md")

    def parse(self, file: RawFile, content: bytes) -> Document:
        return Document(source=file.source, rel_path=file.rel_path, text="md", sections=[])


def _raw(name: str) -> RawFile:
    return RawFile("s", name, "/" + name, 0.0, "h")


def test_router_dispatches_by_extension():
    router = FormatRouter([_FakeMd()])
    doc = router.parse(_raw("a.md"), b"x")
    assert doc.text == "md"


def test_router_unknown_format_raises():
    router = FormatRouter([_FakeMd()])
    with pytest.raises(ValueError, match="不支持"):
        router.parse(_raw("a.unknown"), b"x")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_adapter_router.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/pipeline/adapters/router.py`**

```python
"""适配器路由：按文件扩展名分发到对应适配器。"""

from fin_rag.models import Document, RawFile


class FormatRouter:
    def __init__(self, adapters: list) -> None:
        self.adapters = adapters

    def parse(self, file: RawFile, content: bytes) -> Document:
        for ad in self.adapters:
            if ad.supports(file.rel_path):
                return ad.parse(file, content)
        raise ValueError(f"不支持的文件格式: {file.rel_path}")

    def supports(self, rel_path: str) -> bool:
        return any(ad.supports(rel_path) for ad in self.adapters)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_adapter_router.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/pipeline/adapters/router.py tests/test_adapter_router.py
git commit -m "feat: FormatRouter 适配器路由"
```

---

## Task 9: Chunker（标题树切分 + 父子关系）

**Files:**
- Create: `src/fin_rag/pipeline/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: 写失败测试 `tests/test_chunker.py`**

```python
from fin_rag.models import Document, RawFile, Section
from fin_rag.pipeline.chunker import Chunker

TARGET, OVERLAP = 200, 40


def _doc(sections: list[Section]) -> Document:
    return Document(source="fin-online", rel_path="docs/a.md", text="", sections=sections)


def test_one_node_per_short_section_with_parent_link():
    parent = Section(1, "总览", "短", "总览")
    child = Section(2, "鉴权", "短", "总览/鉴权")
    nodes = Chunker(target_size=TARGET, overlap=OVERLAP).chunk(
        _doc([parent, child]), file_hash="h"
    )
    assert len(nodes) == 2
    child_node = next(n for n in nodes if n.heading_path == "总览/鉴权")
    parent_node = next(n for n in nodes if n.heading_path == "总览")
    assert child_node.parent_node_id == parent_node.node_id


def test_long_section_splits_with_overlap():
    long_text = "字" * (TARGET * 3)
    sec = Section(2, "长节", long_text, "长节")
    nodes = Chunker(target_size=TARGET, overlap=OVERLAP).chunk(
        _doc([sec]), file_hash="h"
    )
    assert len(nodes) > 1
    # 块大小近似 target
    assert all(len(n.text) <= TARGET + OVERLAP for n in nodes)


def test_chunk_index_unique_per_file():
    sec = Section(1, "a", "x", "a")
    sec2 = Section(1, "b", "y", "b")
    nodes = Chunker(target_size=TARGET, overlap=OVERLAP).chunk(
        _doc([sec, sec2]), file_hash="h"
    )
    idx = sorted(n.chunk_index for n in nodes)
    assert idx == list(range(len(nodes)))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/pipeline/chunker.py`**

```python
"""结构化切分：按 Section 切，长节按 target_size 滚窗 + overlap，建立父子关系。"""

from fin_rag.models import Document, Node, make_node

# 标题层级 → 估算每个汉字 ≈ 1.5 token，这里按字符近似控制（M2 可换 tokenizer）
_CHARS_PER_TOKEN = 1.5


class Chunker:
    def __init__(self, target_size: int = 400, overlap: int = 60) -> None:
        self.target_tokens = target_size
        self.overlap_tokens = overlap

    def chunk(self, doc: Document, file_hash: str) -> list[Node]:
        nodes: list[Node] = []
        parent_lookup: dict[str, str] = {}  # heading_path -> node_id
        chunk_index = 0

        for section in doc.sections:
            target_chars = int(self.target_tokens * _CHARS_PER_TOKEN)
            overlap_chars = int(self.overlap_tokens * _CHARS_PER_TOKEN)
            pieces = self._split_text(section.text, target_chars, overlap_chars)
            if not pieces:
                pieces = [section.text]

            parent_id = self._find_parent(section, parent_lookup)
            for piece in pieces:
                node = make_node(
                    source=doc.source,
                    rel_path=doc.rel_path,
                    text=piece,
                    chunk_index=chunk_index,
                    project=doc.source,
                    domain=doc.source,  # 占位，MetadataExtractor 覆盖
                    source_type="doc",
                    doc_type="tech_plan",  # 占位，MetadataExtractor 覆盖
                    heading_path=section.heading_path,
                    file_hash=file_hash,
                    mtime=0.0,
                    parent_node_id=parent_id,
                )
                nodes.append(node)
                parent_lookup[section.heading_path] = node.node_id
                chunk_index += 1
        return nodes

    def _split_text(self, text: str, target: int, overlap: int) -> list[str]:
        text = text.strip()
        if len(text) <= target:
            return [text] if text else []
        pieces: list[str] = []
        step = max(target - overlap, 1)
        i = 0
        while i < len(text):
            piece = text[i : i + target]
            if piece.strip():
                pieces.append(piece.strip())
            i += step
        return pieces

    def _find_parent(self, section, parent_lookup: dict[str, str]):
        parts = section.heading_path.split("/")
        for depth in range(len(parts) - 1, 0, -1):
            ancestor = "/".join(parts[:depth])
            if ancestor in parent_lookup:
                return parent_lookup[ancestor]
        return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/pipeline/chunker.py tests/test_chunker.py
git commit -m "feat: Chunker（标题树切分 + 滚窗overlap + 父子关系）"
```

---

## Task 10: MetadataExtractor（domain/doc_type/产品号/渠道 打标）

**Files:**
- Create: `src/fin_rag/pipeline/metadata.py`
- Test: `tests/test_metadata.py`

- [ ] **Step 1: 写失败测试 `tests/test_metadata.py`**

```python
from fin_rag.config import SourceConfig
from fin_rag.models import make_node
from fin_rag.pipeline.metadata import MetadataExtractor


def _cfg(default_domain="fin-online"):
    return SourceConfig(
        name="fin-online",
        root="/tmp",
        include=["docs/"],
        default_domain=default_domain,
        exclude_patterns=[],
    )


def _node(rel, heading, text=""):
    return make_node(
        source="fin-online", rel_path=rel, text=text or heading, chunk_index=0,
        project="fin-online", domain="fin-online", source_type="doc",
        doc_type="tech_plan", heading_path=heading, file_hash="h", mtime=0.0,
    )


def test_doc_type_inferred_from_directory():
    me = MetadataExtractor()
    n = me.enrich(_node("docs/spec/x.md", "需求"), _cfg())
    assert n.doc_type == "spec"
    n2 = me.enrich(_node("docs/plan/x.md", "方案"), _cfg())
    assert n2.doc_type == "tech_plan"
    n3 = me.enrich(_node(".cursor/rules/core.mdc", "规则"), _cfg())
    assert n3.doc_type == "engineering_rule"
    assert n3.source_type == "cursor_rule"
    n4 = me.enrich(_node(".cursor/memory/special/x.md", "账期"), _cfg())
    assert n4.doc_type == "business_logic"
    assert n4.source_type == "cursor_memory"


def test_domain_falls_back_to_default():
    me = MetadataExtractor()
    n = me.enrich(_node("docs/plan/普通.md", "方案"), _cfg(default_domain="fin-online"))
    assert n.domain == "fin-online"


def test_product_code_and_channel_extracted_from_text():
    me = MetadataExtractor()
    n = me.enrich(
        _node("docs/plan/x.md", "方案", "300007 cucc 联通渠道对接"), _cfg()
    )
    assert n.product_code == "300007"
    assert n.channel == "cucc"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_metadata.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/pipeline/metadata.py`**

```python
"""元数据打标：从路径/标题/正文推断 doc_type/source_type/domain/产品号/渠道。"""

import re
from dataclasses import replace

from fin_rag.config import SourceConfig
from fin_rag.models import Node

# 目录 → doc_type
_DOC_TYPE_BY_DIR = [
    ("docs/plan", "tech_plan"),
    ("docs/spec", "spec"),
    ("docs/channel-optimization", "tech_plan"),
    ("docs/channels", "channel_doc"),
    (".cursor/rules", "engineering_rule"),
    (".cursor/memory/special", "business_logic"),
    (".cursor/memory/requirement", "tech_plan"),
]

# source_type
_SOURCE_TYPE_BY_DIR = [
    (".cursor/rules", "cursor_rule"),
    (".cursor/memory", "cursor_memory"),
    ("docs/channels", "channel_doc"),
]

# 二期才正式纳入的域，这里仅识别打标（首版数据源不含这些目录）
_DOMAIN_KEYWORDS = {
    "shield": ["shield", "盾"],
    "jindun": ["jindun", "金盾"],
    "hmf": ["hmf"],
    "sjf": ["sjf"],
    "bxf": ["bxf"],
    "opdata": ["opdata"],
}

_PRODUCT_RE = re.compile(r"\b(3\d{5}|6\d{5})\b")
_CHANNEL_KEYWORDS = ["cucc", "cmcc", "ctcc", "lan chen", "lanchen", "rong360"]


class MetadataExtractor:
    def enrich(self, node: Node, source: SourceConfig) -> Node:
        doc_type = self._infer_doc_type(node.rel_path) or node.doc_type
        source_type = self._infer_source_type(node.rel_path) or node.source_type
        domain = self._infer_domain(node, source) or source.default_domain
        product_code = self._extract_product(node.text + " " + node.heading_path)
        channel = self._extract_channel(node.text + " " + node.rel_path)
        return replace(
            node,
            doc_type=doc_type,
            source_type=source_type,
            domain=domain,
            product_code=product_code,
            channel=channel,
        )

    def _infer_doc_type(self, rel_path: str) -> str | None:
        low = rel_path.lower()
        for prefix, dt in _DOC_TYPE_BY_DIR:
            if prefix in low:
                return dt
        if "架构" in rel_path:
            return "arch"
        return None

    def _infer_source_type(self, rel_path: str) -> str | None:
        low = rel_path.lower()
        for prefix, st in _SOURCE_TYPE_BY_DIR:
            if prefix in low:
                return st
        return None

    def _infer_domain(self, node: Node, source: SourceConfig) -> str | None:
        haystack = (node.rel_path + " " + node.heading_path + " " + node.text).lower()
        for dom, kws in _DOMAIN_KEYWORDS.items():
            if any(kw in haystack for kw in kws):
                return dom
        return source.default_domain

    def _extract_product(self, text: str) -> str | None:
        m = _PRODUCT_RE.search(text)
        return m.group(1) if m else None

    def _extract_channel(self, text: str) -> str | None:
        low = text.lower()
        for kw in _CHANNEL_KEYWORDS:
            if kw.replace(" ", "") in low.replace(" ", ""):
                return kw.replace(" ", "")
        return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_metadata.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/pipeline/metadata.py tests/test_metadata.py
git commit -m "feat: MetadataExtractor（doc_type/source_type/domain/产品号/渠道 打标）"
```

---

## Task 11: Embedder（智谱 Embedding-3，OpenAI 兼容，可 mock）

**Files:**
- Create: `src/fin_rag/pipeline/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: 写失败测试 `tests/test_embedder.py`**

```python
from unittest.mock import MagicMock, patch

from fin_rag.config import Settings
from fin_rag.pipeline.embedder import OpenAIEmbedder


def _settings():
    return Settings(
        zhipu_api_key="test",
        embedding_model="embedding-3",
        embedding_dim=1024,
        embedding_batch_size=2,
    )


def test_embed_returns_vectors_matching_input_count():
    emb = OpenAIEmbedder(_settings())
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1024), MagicMock(embedding=[0.2] * 1024)]
    )
    with patch.object(emb, "_client", fake_client):
        vecs = emb.embed(["你好", "世界"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 1024


def test_embed_batches_by_batch_size():
    emb = OpenAIEmbedder(_settings())
    fake_client = MagicMock()
    # batch_size=2，3 条文本应分 2 批
    fake_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.0] * 1024)] * 2
    )
    with patch.object(emb, "_client", fake_client):
        emb.embed(["a", "b", "c"])
    assert fake_client.embeddings.create.call_count == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/pipeline/embedder.py`**

```python
"""Embedder：智谱 Embedding-3，OpenAI 兼容接口，按 batch 分批，dimensions 指定维度。"""

from openai import OpenAI

from fin_rag.config import Settings


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.zhipu_api_key,
            base_url=settings.zhipu_base_url,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_vecs: list[list[float]] = []
        batch = self.settings.embedding_batch_size
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            resp = self._client.embeddings.create(
                model=self.settings.embedding_model,
                input=chunk,
                dimensions=self.settings.embedding_dim,  # 智谱 Embedding-3 指定维度
            )
            all_vecs.extend([d.embedding for d in resp.data])
        return all_vecs
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_embedder.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/pipeline/embedder.py tests/test_embedder.py
git commit -m "feat: Embedder（智谱 Embedding-3，dimensions 指定维度，分批向量化）"
```

---

## Task 12: VectorStore（Qdrant dense upsert 幂等）

**Files:**
- Create: `src/fin_rag/store/__init__.py`
- Create: `src/fin_rag/store/vectorstore.py`
- Test: `tests/test_vectorstore.py`

- [ ] **Step 1: 写失败测试 `tests/test_vectorstore.py`**

```python
from unittest.mock import MagicMock

from fin_rag.config import Settings
from fin_rag.models import make_node
from fin_rag.store.vectorstore import QdrantVectorStore


def _settings():
    return Settings(
        zhipu_api_key="test",
        qdrant_collection="fin_rag",
        embedding_dim=1024,
    )


def _node(i):
    return make_node(
        source="fin-online", rel_path="docs/a.md", text=f"t{i}", chunk_index=i,
        project="fin-online", domain="fin-online", source_type="doc",
        doc_type="tech_plan", heading_path="h", file_hash="hash", mtime=0.0,
    )


def test_ensure_collection_creates_with_dim():
    store = QdrantVectorStore(_settings())
    fake = MagicMock()
    fake.collection_exists.return_value = False
    store._client = fake
    store.ensure_collection()
    fake.create_collection.assert_called_once()
    args, kwargs = fake.create_collection.call_args
    assert kwargs["vectors_config"]["size"] == 1024


def test_upsert_uses_node_id_as_point_id():
    store = QdrantVectorStore(_settings())
    fake = MagicMock()
    store._client = fake
    nodes = [_node(0), _node(1)]
    vecs = [[0.1] * 1024, [0.2] * 1024]
    store.upsert(nodes, vecs)
    fake.upsert.assert_called_once()
    _, kwargs = fake.upsert.call_args
    point_ids = [p.id for p in kwargs["points"]]
    assert point_ids == [n.node_id for n in nodes]
    # payload 含 domain/doc_type/溯源
    payloads = [p.payload for p in kwargs["points"]]
    assert "domain" in payloads[0]
    assert "heading_path" in payloads[0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_vectorstore.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/store/__init__.py`（空）**

```python
```

- [ ] **Step 4: 写 `src/fin_rag/store/vectorstore.py`**

```python
"""Qdrant 向量库封装：建集合（dense）、幂等 upsert（point id = node_id）。"""

from qdrant_client import QdrantClient
from qdrant_client.http import models

from fin_rag.config import Settings
from fin_rag.models import Node


class QdrantVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = QdrantClient(url=settings.qdrant_url)

    def ensure_collection(self) -> None:
        name = self.settings.qdrant_collection
        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dim, distance=models.Distance.COSINE
            ),
        )

    def upsert(self, nodes: list[Node], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=node.node_id,
                vector=vec,
                payload=self._payload(node),
            )
            for node, vec in zip(nodes, vectors)
        ]
        self._client.upsert(
            collection_name=self.settings.qdrant_collection, points=points
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

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_vectorstore.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: Commit**

```bash
git add src/fin_rag/store tests/test_vectorstore.py
git commit -m "feat: QdrantVectorStore（建集合 + 幂等 upsert + 溯源 payload）"
```

---

## Task 13: 管道编排 + CLI（全量建索引，端到端可运行）

**Files:**
- Create: `src/fin_rag/pipeline/pipeline.py`
- Create: `src/fin_rag/cli.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写失败测试 `tests/test_pipeline.py`（用 fake fixtures + mock embed/store）**

```python
from pathlib import Path
from unittest.mock import MagicMock

from fin_rag.config import SourceConfig
from fin_rag.pipeline.adapters.router import FormatRouter
from fin_rag.pipeline.adapters.markdown import MarkdownAdapter
from fin_rag.pipeline.adapters.mdc import MdcAdapter
from fin_rag.pipeline.chunker import Chunker
from fin_rag.pipeline.datasource import FilesystemDataSource
from fin_rag.pipeline.metadata import MetadataExtractor
from fin_rag.pipeline.pipeline import IndexPipeline

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_project"


def _cfg():
    return SourceConfig(
        name="sample", root=str(FIXTURE), include=["docs/", ".cursor/"],
        exclude_patterns=["*bxf*", "docs/testcase/", ".DS_Store"],
        default_domain="sample",
    )


def test_pipeline_runs_end_to_end_with_mocked_embed_and_store():
    router = FormatRouter([MdcAdapter(), MarkdownAdapter()])
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 1024]
    store = MagicMock()

    pipe = IndexPipeline(
        sources=[_cfg()],
        datasource_factory=FilesystemDataSource,
        router=router,
        chunker=Chunker(target_size=400, overlap=60),
        extractor=MetadataExtractor(),
        embedder=embedder,
        store=store,
    )
    count = pipe.run()

    assert count > 0
    store.ensure_collection.assert_called_once()
    store.upsert.assert_called()
    # 传给 store 的 node payload 含正确 domain
    _, kwargs = store.upsert.call_args
    payloads = [p.payload for p in kwargs["points"]]
    assert all(p["domain"] == "sample" for p in payloads)
```

> 注：此测试依赖 Task 4 创建的 `tests/fixtures/sample_project`，并验证 bxf/testcase 已被排除（不会被解析）。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/pipeline/pipeline.py`**

```python
"""索引管道编排：采集 → 适配 → 切分 → 打标 → 向量化 → 入库。"""

from fin_rag.config import SourceConfig
from fin_rag.models import Node
from fin_rag.pipeline.adapters.router import FormatRouter
from fin_rag.pipeline.chunker import Chunker
from fin_rag.pipeline.datasource import DataSource, FilesystemDataSource
from fin_rag.pipeline.metadata import MetadataExtractor


class IndexPipeline:
    def __init__(
        self,
        sources: list[SourceConfig],
        datasource_factory,
        router: FormatRouter,
        chunker: Chunker,
        extractor: MetadataExtractor,
        embedder,
        store,
    ) -> None:
        self.sources = sources
        self.datasource_factory = datasource_factory
        self.router = router
        self.chunker = chunker
        self.extractor = extractor
        self.embedder = embedder
        self.store = store

    def run(self) -> int:
        self.store.ensure_collection()
        total = 0
        for cfg in self.sources:
            ds: DataSource = self.datasource_factory(cfg)
            for raw in ds.list_files():
                if not self.router.supports(raw.rel_path):
                    continue
                content = ds.read(raw)
                doc = self.router.parse(raw, content)
                nodes = self.chunker.chunk(doc, file_hash=raw.file_hash)
                # 用真实 mtime 回填（chunk 默认 0.0）
                nodes = [
                    self.extractor.enrich(
                        _with_mtime(n, raw.mtime), cfg
                    )
                    for n in nodes
                ]
                if not nodes:
                    continue
                vectors = self.embedder.embed([n.text for n in nodes])
                self.store.upsert(nodes, vectors)
                total += len(nodes)
        return total


def _with_mtime(node: Node, mtime: float) -> Node:
    from dataclasses import replace
    return replace(node, mtime=mtime)
```

- [ ] **Step 4: 写 `src/fin_rag/cli.py`**

```python
"""CLI 入口：fin-rag index 全量建索引。"""

import argparse
import logging
import sys
from pathlib import Path

from fin_rag.config import load_settings, load_sources
from fin_rag.pipeline.adapters.csv_tabular import (  # noqa: F401  占位，下面 Step 替换
)
```

> ⚠️ 上面的 `cli.py` import 路径需修正——`tabular.py` 里的类是 `CsvAdapter/XlsxAdapter`。Step 5 给出正确完整版。

- [ ] **Step 5: 用正确完整版覆盖 `src/fin_rag/cli.py`**

```python
"""CLI 入口：fin-rag index 全量建索引。"""

import argparse
import logging
import sys
from pathlib import Path

from fin_rag.config import load_settings, load_sources
from fin_rag.pipeline.adapters.markdown import MarkdownAdapter
from fin_rag.pipeline.adapters.mdc import MdcAdapter
from fin_rag.pipeline.adapters.router import FormatRouter
from fin_rag.pipeline.adapters.tabular import CsvAdapter, XlsxAdapter
from fin_rag.pipeline.chunker import Chunker
from fin_rag.pipeline.datasource import FilesystemDataSource
from fin_rag.pipeline.embedder import OpenAIEmbedder
from fin_rag.pipeline.metadata import MetadataExtractor
from fin_rag.pipeline.pipeline import IndexPipeline
from fin_rag.store.vectorstore import QdrantVectorStore

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
log = logging.getLogger("fin-rag")


def _build_pipeline():
    settings = load_settings(CONFIG_DIR / "settings.yaml")
    sources = load_sources(CONFIG_DIR / "sources.yaml")
    router = FormatRouter([MdcAdapter(), MarkdownAdapter(), XlsxAdapter(), CsvAdapter()])
    return IndexPipeline(
        sources=sources,
        datasource_factory=FilesystemDataSource,
        router=router,
        chunker=Chunker(target_size=400, overlap=60),
        extractor=MetadataExtractor(),
        embedder=OpenAIEmbedder(settings),
        store=QdrantVectorStore(settings),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="fin-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="全量构建索引")
    args = parser.parse_args(argv)

    if args.cmd == "index":
        try:
            count = _build_pipeline().run()
            log.info("索引构建完成，共 %d 个切块", count)
            return 0
        except Exception as exc:  # 全链路兜底
            log.error("索引构建失败：%s", exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS（1 passed）

- [ ] **Step 7: 运行全量测试 + 覆盖率检查**

Run: `pytest -v`
Expected: 全部 PASS，`fin_rag` 覆盖率 ≥ 80%

- [ ] **Step 8: 端到端手动验证（需 Qdrant + 真实 API key）**

Run:
```bash
cd /Users/wangfei/yulore/fin-rag
cp .env.example .env  # 填入真实 ZHIPU_API_KEY
docker compose up -d qdrant
source .venv/bin/activate
fin-rag index
```
Expected: 日志输出「索引构建完成，共 N 个切块」，N > 0（fin-online 约 200+ 切块，opdata 约 10 切块，bxf/testcase/review 已排除）。

用 Qdrant Dashboard（http://localhost:6333/dashboard）或以下命令抽样核验：
```bash
python -c "from qdrant_client import QdrantClient; c=QdrantClient(url='http://localhost:6333'); r=c.scroll('fin_rag', limit=3); print([(p.payload['domain'], p.payload['doc_type'], p.payload['file_path']) for p in r[0]])"
```
Expected: 能看到 `domain=fin-online`/`opdata`，`doc_type` 含 `tech_plan/spec/business_logic/engineering_rule` 等，无 bxf/testcase 路径。

- [ ] **Step 9: Commit**

```bash
git add src/fin_rag/pipeline/pipeline.py src/fin_rag/cli.py tests/test_pipeline.py
git commit -m "feat: 索引管道编排 + CLI 全量建索引（端到端可运行）"
```

---

## 完成标准（M0-M1）

- [ ] 全部 13 个任务测试通过，`pytest` 覆盖率 ≥ 80%
- [ ] `fin-rag index` 能在真实数据上跑通，产出非空索引
- [ ] 抽样核验：bxf/testcase/review 路径不出现在索引中；domain/doc_type 标签正确
- [ ] 元数据 payload 完整（含溯源字段 file_path/heading_path）

## 下一步

M0-M1 完成、索引验证通过后，进入**第二份计划（M2-M5）**：混合检索 + 重排 → 防幻觉三道闸 + 溯源 + DeepSeek 问答编排 → Gradio ChatInterface 前端 → Golden Set 评测 → 增量同步 + Docker 部署。
