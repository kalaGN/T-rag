# fin-rag M4 实现计划：Gradio 前端（多轮流式 + 溯源面板 + 反馈）

> **已废弃（2026-06-25）**：其后端接口依赖旧 QAChain，需在 LlamaIndex Workflow 实施计划确定后重新生成。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 M3 的 `QAChain` 套上 **Gradio Blocks** Web 前端：多轮流式聊天 + 引用溯源面板 + 业务域筛选 + 👍👎 反馈（落 SQLite）。`fin-rag serve` 启动，团队浏览器访问。

**Architecture:** `WebApp`（Gradio Blocks）调用 `QAChain.ask_stream`（新增流式生成 + 末尾引用校验③）。`MetadataStore`（SQLite）append-only 记录问答历史与反馈（并预埋同步状态表，供 M5 复用）。

**Tech Stack:** 新增 `gradio>=4.0`；`sqlite3` 标准库。其余复用 M0-M3。

**关键设计：**
- **流式与防幻觉的协调**：`ask_stream` 流式 yield token（用户即时看到逐字），生成完成后做引用校验③，把 `QAResult`（置信度/引用/闸结果）作为 `StreamChunk(done=True)` 最后吐出；前端据此渲染溯源面板。门控①在生成前拦截（直接拒答，不流式）。
- **Gradio 流式**：`chat_fn` 是 generator，`yield` 累积的 `chatbot`（`type="messages"`）逐步更新；流式结束后 `yield` 溯源面板。
- **反馈**：溯源面板显示本条 `qa_id`，用户填评价点提交 → `record_feedback`。简化版（不做每条内联按钮，避免 Gradio 动态组件复杂度）。

**接口约定（沿用）：** M3 `QAResult`/`Citation`/`QAChain`、M2 `RankedChunk`/`Retriever`、M0-M1 `Settings`。

---

## Task 1: LLMClient.stream_chat + QAChain.ask_stream

**Files:**
- Modify: `src/fin_rag/qa/llm_client.py`（新增 `stream_chat`）
- Modify: `src/fin_rag/qa/chain.py`（新增 `StreamChunk` + `ask_stream`）
- Test: `tests/test_qa_stream.py`

- [ ] **Step 1: 写失败测试 `tests/test_qa_stream.py`**

```python
from unittest.mock import MagicMock

from fin_rag.config import Settings
from fin_rag.qa.chain import QAChain, StreamChunk
from fin_rag.qa.context import Citation
from fin_rag.qa.guardrails import GateResult
from fin_rag.qa.llm_client import LLMClient


def _settings():
    return Settings(zhipu_api_key="x", deepseek_api_key="sk")


def _chain(gate_passed=True, validator_valid=True):
    retriever = MagicMock()
    from fin_rag.retrieval.types import RankedChunk
    retriever.search.return_value = [RankedChunk(
        node_id="n", text="账期 300007", score=0.8, project="fin-online",
        domain="fin-online", source_type="doc", doc_type="business_logic",
        product_code="300007", channel=None, heading_path="账期",
        file_path="docs/a.md", parent_node_id=None)] if gate_passed else []
    llm = MagicMock()
    llm.stream_chat.return_value = iter(["账", "期", "见 [1]。"])
    prompt = MagicMock(); prompt.build_system.return_value = "S"; prompt.build_user.return_value = "U"
    ctx = MagicMock()
    ctx.build.return_value = ("[1] 账期 300007", [Citation(1, "[d|f|§h]", "f", "h", "账期 300007")])
    gate = MagicMock()
    gate.decide.return_value = GateResult(gate_passed, "" if gate_passed else "无召回")
    validator = MagicMock()
    validator.validate.return_value = MagicMock(valid=validator_valid, invalid_refs=[], suspect_products=[])
    router = MagicMock(); router.route.return_value = "document"
    return QAChain(retriever, llm, prompt, ctx, gate, validator, router)


def test_llm_stream_chat_yields_deltas():
    client = LLMClient(_settings())
    fake = MagicMock()
    d1 = MagicMock(); d1.choices = [MagicMock(delta=MagicMock(content="你"))]
    d2 = MagicMock(); d2.choices = [MagicMock(delta=MagicMock(content="好"))]
    d3 = MagicMock(); d3.choices = [MagicMock(delta=MagicMock(content=None))]
    fake.chat.completions.create.return_value = iter([d1, d2, d3])
    client._client = fake
    out = list(client.stream_chat(system="s", history=[], user="u"))
    assert out == ["你", "好"]
    _, kw = fake.chat.completions.create.call_args
    assert kw["stream"] is True


def test_ask_stream_yields_deltas_then_done_result():
    chain = _chain()
    events = list(chain.ask_stream("账期怎么切"))
    deltas = [e.delta for e in events if not e.done]
    done = [e for e in events if e.done]
    assert "".join(deltas) == "账期见 [1]。"
    assert len(done) == 1
    assert done[0].result is not None
    assert done[0].result.answer == "账期见 [1]。"
    assert done[0].result.confidence in ("high", "low")


def test_ask_stream_gates_before_streaming():
    chain = _chain(gate_passed=False)
    events = list(chain.ask_stream("无关问题"))
    # 门控触发：只吐一个 done（拒答），无 delta
    deltas = [e for e in events if e.delta]
    done = [e for e in events if e.done]
    assert deltas == []
    assert len(done) == 1
    assert done[0].result.gated is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_qa_stream.py -v`
Expected: FAIL（`AttributeError: 'LLMClient' object has no attribute 'stream_chat'`）

- [ ] **Step 3: 在 `src/fin_rag/qa/llm_client.py` 末尾追加 `stream_chat`**

```python
    def stream_chat(self, system: str, history: list[dict], user: str):
        """流式对话：逐 token yield delta（供 Gradio 流式显示）。"""
        messages = [{"role": "system", "content": system}] + list(history) + [
            {"role": "user", "content": user}
        ]
        stream = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.1, stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

- [ ] **Step 4: 在 `src/fin_rag/qa/chain.py` 追加 `StreamChunk` 与 `ask_stream`**

文件顶部 import 区补 `from typing import Iterator, Optional`（若已有 `Optional` 则只补 `Iterator`）。在 `QAResult` 定义之后、`QAChain` 之前追加：

```python
@dataclass(frozen=True)
class StreamChunk:
    delta: str                         # 增量文本（done=False 时）
    done: bool                         # 是否终结
    result: Optional[QAResult] = None  # done=True 时的完整结果（含校验）
```

在 `QAChain` 类内（`ask` 方法之后）追加 `ask_stream`：

```python
    def ask_stream(
        self,
        query: str,
        history: Optional[list[dict]] = None,
        domains: Optional[list[str]] = None,
    ) -> Iterator[StreamChunk]:
        """流式问答：先门控①，通过后流式生成，末尾校验③ 并吐 QAResult。"""
        if self._router.route(query) != "document":
            yield StreamChunk("", True, QAResult(
                answer="数据查询二期支持，当前仅文档问答。",
                citations=[], gated=False, gate_reason="data_route",
                citation_check=None, confidence="low"))
            return

        chunks = self._retriever.search(query, domains=domains)
        gate_result = self._gate.decide(chunks)
        if not gate_result.passed:
            yield StreamChunk("", True, QAResult(
                answer=GATE_REFUSAL_TEMPLATE, citations=[], gated=True,
                gate_reason=gate_result.reason, citation_check=None, confidence="low"))
            return

        context, citations = self._context.build(chunks)
        system = self._prompt.build_system(domain=self._domain, extra_rules=self._extra_rules)
        user = self._prompt.build_user(query=query, context=context)

        parts: list[str] = []
        for delta in self._llm.stream_chat(system=system, history=history or [], user=user):
            parts.append(delta)
            yield StreamChunk(delta=delta, done=False)

        answer = "".join(parts)
        check = self._validator.validate(answer, citations)
        yield StreamChunk("", True, QAResult(
            answer=answer, citations=citations, gated=False, gate_reason="",
            citation_check=check, confidence="high" if check.valid else "low"))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_qa_stream.py -v`
Expected: PASS（4 passed）

- [ ] **Step 6: Commit**

```bash
git add src/fin_rag/qa/llm_client.py src/fin_rag/qa/chain.py tests/test_qa_stream.py
git commit -m "feat: LLM 流式 + QAChain.ask_stream（流式生成 + 末尾引用校验）"
```

---

## Task 2: MetadataStore（SQLite，问答历史 + 反馈 + 同步状态）

**Files:**
- Create: `src/fin_rag/store/metadata_store.py`
- Test: `tests/test_metadata_store.py`

> 预埋 `sync_state` 表，M5 增量同步直接复用，不重复建表。

- [ ] **Step 1: 写失败测试 `tests/test_metadata_store.py`**

```python
from fin_rag.qa.context import Citation
from fin_rag.store.metadata_store import MetadataStore


def _cite(i=1):
    return Citation(id=i, label="[fin-online|docs/a.md|§账期]", file_path="docs/a.md",
                    heading_path="账期", text="正文")


def test_record_and_list_qa_history(tmp_path):
    store = MetadataStore(str(tmp_path / "meta.db"))
    store.record_qa("q1", "账期怎么切", "见 [1]。", [_cite(1)], "high")
    store.record_qa("q2", "另一个问题", "答2", [_cite(2)], "low")

    hist = store.list_history(limit=10)
    assert len(hist) == 2
    assert hist[0]["query"] in ("账期怎么切", "另一个问题")


def test_record_feedback(tmp_path):
    store = MetadataStore(str(tmp_path / "meta.db"))
    store.record_qa("q1", "q", "a", [], "high")
    store.record_feedback("q1", "up")
    store.record_feedback("q1", "down")
    # 反馈 append-only，两条都保留
    assert store.count_feedback("q1") == 2


def test_sync_state_roundtrip(tmp_path):
    store = MetadataStore(str(tmp_path / "meta.db"))
    assert store.get_sync_state("fin-online", "docs/a.md") is None
    store.set_sync_state("fin-online", "docs/a.md", "hash1", 1.0)
    row = store.get_sync_state("fin-online", "docs/a.md")
    assert row["file_hash"] == "hash1"
    # 更新（覆盖）
    store.set_sync_state("fin-online", "docs/a.md", "hash2", 2.0)
    assert store.get_sync_state("fin-online", "docs/a.md")["file_hash"] == "hash2"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_metadata_store.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/store/metadata_store.py`**

```python
"""SQLite 元数据存储：问答历史 + 反馈 + 同步状态，全部 append-only。"""

import json
import sqlite3
import time
from dataclasses import asdict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_history (
    qa_id TEXT PRIMARY KEY, ts REAL, query TEXT, answer TEXT,
    citations TEXT, confidence TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, qa_id TEXT, vote TEXT
);
CREATE TABLE IF NOT EXISTS sync_state (
    source TEXT, rel_path TEXT, file_hash TEXT, mtime REAL,
    PRIMARY KEY (source, rel_path)
);
"""


class MetadataStore:
    def __init__(self, path: str = "data/fin_rag_meta.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def record_qa(self, qa_id, query, answer, citations, confidence) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO qa_history VALUES (?,?,?,?,?,?)",
            (qa_id, time.time(), query, answer,
             json.dumps([asdict(c) for c in citations], ensure_ascii=False), confidence),
        )
        self.conn.commit()

    def list_history(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT qa_id, ts, query, answer, citations, confidence "
            "FROM qa_history ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def record_feedback(self, qa_id, vote) -> None:
        self.conn.execute(
            "INSERT INTO feedback (ts, qa_id, vote) VALUES (?,?,?)",
            (time.time(), qa_id, vote),
        )
        self.conn.commit()

    def count_feedback(self, qa_id) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM feedback WHERE qa_id=?", (qa_id,)
        ).fetchone()
        return int(row["n"])

    def get_sync_state(self, source, rel_path) -> dict | None:
        row = self.conn.execute(
            "SELECT source, rel_path, file_hash, mtime FROM sync_state "
            "WHERE source=? AND rel_path=?", (source, rel_path)
        ).fetchone()
        return dict(row) if row else None

    def set_sync_state(self, source, rel_path, file_hash, mtime) -> None:
        self.conn.execute(
            "INSERT INTO sync_state (source, rel_path, file_hash, mtime) VALUES (?,?,?,?) "
            "ON CONFLICT(source, rel_path) DO UPDATE SET file_hash=?, mtime=?",
            (source, rel_path, file_hash, mtime, file_hash, mtime),
        )
        self.conn.commit()

    def all_sync_state(self, source) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT rel_path, file_hash, mtime FROM sync_state WHERE source=?", (source,)
        ).fetchall()
        return {r["rel_path"]: {"file_hash": r["file_hash"], "mtime": r["mtime"]} for r in rows}

    def delete_sync_state(self, source, rel_path) -> None:
        self.conn.execute(
            "DELETE FROM sync_state WHERE source=? AND rel_path=?", (source, rel_path)
        )
        self.conn.commit()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_metadata_store.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/store/metadata_store.py tests/test_metadata_store.py
git commit -m "feat: MetadataStore（SQLite 问答历史/反馈/同步状态，append-only）"
```

---

## Task 3: WebApp（Gradio Blocks：多轮流式 + 溯源面板 + 反馈）

**Files:**
- Create: `src/fin_rag/web/__init__.py`
- Create: `src/fin_rag/web/app.py`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: 写失败测试 `tests/test_web_app.py`（测 chat 逻辑，不测 Gradio 渲染）**

```python
from unittest.mock import MagicMock

from fin_rag.qa.chain import QAResult, StreamChunk
from fin_rag.qa.context import Citation
from fin_rag.web.app import WebApp


def _app():
    qa = MagicMock()
    store = MagicMock()
    return WebApp(qa_chain=qa, store=store, domains=["全部", "fin-online", "opdata"])


def test_chat_streams_assistant_text_and_final_citations():
    app = _app()
    app.qa_chain.ask_stream.return_value = iter([
        StreamChunk("账", False),
        StreamChunk("期", False),
        StreamChunk("", True, QAResult(
            answer="账期", citations=[Citation(1, "[fin-online|docs/a.md|§账期]", "docs/a.md", "账期", "正文")],
            gated=False, gate_reason="", citation_check=None, confidence="high")),
    ])
    outputs = list(app.chat("账期怎么切", [], "全部"))

    final_chatbot, final_citations = outputs[-1]
    last_msg = final_chatbot[-1]
    assert last_msg["role"] == "assistant"
    assert last_msg["content"] == "账期"
    assert "fin-online|docs/a.md" in final_citations
    app.store.record_qa.assert_called_once()


def test_chat_passes_selected_domain_to_chain():
    app = _app()
    app.qa_chain.ask_stream.return_value = iter([StreamChunk("", True, QAResult(
        answer="x", citations=[], gated=False, gate_reason="", citation_check=None, confidence="high"))])
    list(app.chat("q", [], "opdata"))
    _, kw = app.qa_chain.ask_stream.call_args
    assert kw["domains"] == ["opdata"]


def test_chat_none_domain_means_all():
    app = _app()
    app.qa_chain.ask_stream.return_value = iter([StreamChunk("", True, QAResult(
        answer="x", citations=[], gated=False, gate_reason="", citation_check=None, confidence="high"))])
    list(app.chat("q", [], "全部"))
    _, kw = app.qa_chain.ask_stream.call_args
    assert kw["domains"] is None


def test_render_citations_includes_confidence_and_labels():
    app = _app()
    md = app._render_citations(QAResult(
        answer="a", citations=[Citation(1, "[d|f|§h]", "f", "h", "t")],
        gated=False, gate_reason="", citation_check=None, confidence="low"))
    assert "low" in md and "[d|f|§h]" in md
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_web_app.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/web/__init__.py`（空）**

```python
```

- [ ] **Step 4: 写 `src/fin_rag/web/app.py`**

```python
"""Gradio Blocks 前端：多轮流式问答 + 引用溯源面板 + 业务域筛选 + 反馈。"""

import uuid

import gradio as gr

from fin_rag.qa.chain import QAChain
from fin_rag.store.metadata_store import MetadataStore


class WebApp:
    def __init__(self, qa_chain: QAChain, store: MetadataStore, domains: list[str]) -> None:
        self.qa_chain = qa_chain
        self.store = store
        self.domains = domains

    def chat(self, message: str, history: list, domain: str):
        """Gradio streaming generator：yield (chatbot, citations)。"""
        domains = None if domain in (None, "全部") else [domain]
        prior = [{"role": m["role"], "content": m["content"]} for m in (history or [])]

        working = list(history or []) + [{"role": "user", "content": message}]
        working.append({"role": "assistant", "content": ""})
        assistant_idx = len(working) - 1

        answer_parts: list[str] = []
        final = None
        for ev in self.qa_chain.ask_stream(message, history=prior, domains=domains):
            if ev.delta:
                answer_parts.append(ev.delta)
                working[assistant_idx]["content"] = "".join(answer_parts)
                yield working, None
            if ev.done and ev.result is not None:
                final = ev.result

        if final is not None:
            qa_id = str(uuid.uuid4())
            self.store.record_qa(qa_id, message, final.answer, final.citations, final.confidence)
            yield working, self._render_citations(final, qa_id)

    def _render_citations(self, result, qa_id: str = "") -> str:
        head = f"**置信度：{result.confidence}**"
        if result.gated:
            head += f" · 拒答（{result.gate_reason}）"
        lines = [head]
        for c in result.citations:
            lines.append(f"- [{c.id}] {c.label}")
        if qa_id:
            lines.append("")
            lines.append(f"`qa_id={qa_id}`（反馈请在下方填写评价并提交）")
        return "\n".join(lines)

    def feedback(self, qa_id: str, vote: str) -> str:
        if not qa_id or vote not in ("up", "down"):
            return "请填 qa_id 并选 up/down"
        self.store.record_feedback(qa_id.strip(), vote)
        return f"已记录反馈：{vote}"

    def build(self):
        with gr.Blocks(title="fin-rag 金融知识问答") as demo:
            gr.Markdown("# fin-rag · 金融/运营知识问答（防幻觉 · 全程溯源）")
            with gr.Row():
                domain = gr.Dropdown(choices=self.domains, value="全部", label="业务域筛选")
            chatbot = gr.Chatbot(type="messages", height=460, label="对话")
            citations = gr.Markdown(label="引用溯源 / 置信度")
            with gr.Row():
                msg = gr.Textbox(placeholder="例如：账期切换逻辑是什么？", scale=8, label="问题")
                send = gr.Button("发送", variant="primary", scale=1)
            with gr.Row():
                fb_id = gr.Textbox(label="qa_id", scale=4)
                fb_vote = gr.Radio(["up", "down"], label="反馈", scale=2)
                fb_btn = gr.Button("提交反馈", scale=1)
                fb_out = gr.Markdown()

            send.click(self.chat, [msg, chatbot, domain], [chatbot, citations])
            msg.submit(self.chat, [msg, chatbot, domain], [chatbot, citations])
            fb_btn.click(self.feedback, [fb_id, fb_vote], [fb_out])
        return demo

    def launch(self, **kwargs):
        self.build().launch(**kwargs)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_web_app.py -v`
Expected: PASS（4 passed）

- [ ] **Step 6: Commit**

```bash
git add src/fin_rag/web/__init__.py src/fin_rag/web/app.py tests/test_web_app.py
git commit -m "feat: WebApp（Gradio Blocks 多轮流式 + 溯源面板 + 反馈）"
```

---

## Task 4: 依赖 + CLI serve + 端到端验证

**Files:**
- Modify: `pyproject.toml`（加 `gradio`）
- Modify: `src/fin_rag/cli.py`（新增 `serve` + 工厂）
- Modify: `config/settings.yaml`（新增 `web` 段）
- Test: `tests/test_cli_serve.py`

- [ ] **Step 1: 写失败测试 `tests/test_cli_serve.py`（mock 工厂 + launch，不真起服务）**

```python
from unittest.mock import MagicMock, patch

from fin_rag.cli import main


def test_cli_serve_builds_app_and_launches():
    fake_app = MagicMock()
    with patch("fin_rag.cli._build_web_app") as mk:
        mk.return_value = fake_app
        code = main(["serve", "--port", "7860"])
    assert code == 0
    mk.assert_called_once()
    fake_app.launch.assert_called_once()
    _, kw = fake_app.launch.call_args
    assert kw["server_port"] == 7860


def test_cli_serve_default_port():
    fake_app = MagicMock()
    with patch("fin_rag.cli._build_web_app") as mk:
        mk.return_value = fake_app
        main(["serve"])
    _, kw = fake_app.launch.call_args
    assert kw["server_port"] == 7860
    assert kw["server_name"] == "0.0.0.0"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cli_serve.py -v`
Expected: FAIL（`_build_web_app` 不存在）

- [ ] **Step 3: 改 `pyproject.toml`，dependencies 列表加 gradio**

在 `dependencies = [...]` 中追加一行：

```toml
    "gradio>=4.0",
```

- [ ] **Step 4: 改 `config/settings.yaml` 追加 `web` 段**

```yaml
web:
  server_name: "0.0.0.0"
  server_port: 7860
  domains: ["全部", "fin-online", "opdata"]
```

- [ ] **Step 5: 改 `src/fin_rag/config.py`（`Settings` 加 web 字段 + `load_settings` 回填）**

在 `Settings` 类（`retrieval_score_gate` 之后）追加：

```python
    # Web 前端（M4）
    web_server_name: str = "0.0.0.0"
    web_server_port: int = 7860
    web_domains: list = ["全部", "fin-online", "opdata"]
```

在 `load_settings` 内追加读取与回填：

```python
    web = raw.get("web", {})
```
```python
        web_server_name=web.get("server_name", "0.0.0.0"),
        web_server_port=web.get("server_port", 7860),
        web_domains=list(web.get("domains", ["全部", "fin-online", "opdata"])),
```

- [ ] **Step 6: 改 `src/fin_rag/cli.py`（serve 子命令 + WebApp 工厂）**

import 区追加（`load_settings`/QA 链相关已在 M3 import）：

```python
from fin_rag.store.metadata_store import MetadataStore
from fin_rag.web.app import WebApp
```

在 `_build_qa_chain()`（M3 新增）之后追加：

```python
def _build_web_app():
    settings = load_settings(CONFIG_DIR / "settings.yaml")
    qa_chain = _build_qa_chain()
    store = MetadataStore()
    return WebApp(qa_chain=qa_chain, store=store, domains=settings.web_domains)
```

在 `main()` 的 subparser 注册区（`ask` 之后）追加：

```python
    serve_p = sub.add_parser("serve", help="启动 Gradio Web 服务")
    serve_p.add_argument("--port", type=int, default=None, help="端口（默认 7860）")
    serve_p.add_argument("--host", default=None, help="监听地址（默认 0.0.0.0）")
```

在 `if args.cmd == "ask":` 分支之后追加：

```python
    if args.cmd == "serve":
        try:
            settings = load_settings(CONFIG_DIR / "settings.yaml")
            app = _build_web_app()
            app.launch(
                server_name=args.host or settings.web_server_name,
                server_port=args.port or settings.web_server_port,
            )
            return 0
        except Exception as exc:
            log.error("启动失败：%s", exc)
            return 1
```

- [ ] **Step 7: 安装新依赖并运行全量测试**

Run:
```bash
pip install -e ".[dev]"
pytest -v
```
Expected: 全部 PASS，`fin_rag` 覆盖率 ≥ 80%

- [ ] **Step 8: 端到端手动验证**

```bash
cd /Users/wangfei/yulore/fin-rag
docker compose up -d qdrant
source .venv/bin/activate
fin-rag index
fin-rag serve
```
浏览器开 `http://<服务器IP>:7860`：
- 多轮对话：先问"账期切换逻辑"，再问"那它的鉴权呢"（测多轮历史）。
- 业务域筛选：选 `opdata` 后问渠道问题，只召回 opdata。
- 溯源面板：每条回答下方显示 `[n] [domain|文件|§章节]` 与置信度。
- 反馈：复制溯源面板里的 `qa_id`，选 up/down 提交，提示"已记录"。
- 防幻觉闸①：问库里没有的内容，应显示「知识库未覆盖…」+ `拒答` 标记，**不流式、不调 LLM**。

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml config/settings.yaml src/fin_rag/config.py src/fin_rag/cli.py tests/test_cli_serve.py
git commit -m "feat: CLI serve 启动 Gradio Web 服务（含 web 配置 + 依赖）"
```

---

## 完成标准（M4）

- [ ] 全部 4 个任务测试通过，`pytest` 覆盖率 ≥ 80%
- [ ] `fin-rag serve` 启动 Gradio，浏览器可多轮流式问答
- [ ] 流式逐字显示；门控①在流式前拦截（拒答不流式）
- [ ] 溯源面板显示引用 + 置信度；反馈落 SQLite
- [ ] 业务域筛选生效（选 opdata 只召回 opdata）

## 下一步

M4 完成后，进入**最后一份计划（M5）**：Golden Set 评测（召回命中率/引用准确率/拒答准确率/忠实度）→ 增量同步（hash+mtime 幂等，复用 `MetadataStore.sync_state`）→ Dockerfile + docker-compose 部署。
