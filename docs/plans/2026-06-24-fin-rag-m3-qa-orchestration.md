# fin-rag M3 实现计划：防幻觉三道闸 + 溯源 + DeepSeek 问答编排

> **已废弃（2026-06-25）**：本计划基于自建 QAChain，与当前 LlamaIndex Workflow/CitationQueryEngine 方案不兼容，请勿执行。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 M2 的 `Retriever` 输出接入 **DeepSeek** LLM，实现端到端问答：输入 query + 历史 → 路由 → 检索 → **检索门控(①)** → 上下文组装(带 `[n]` 引用) → 系统提示(防幻觉硬约束 ②) → 生成 → **引用事后校验(③)** → 输出带溯源、置信度标记的 `QAResult`。

**Architecture:** `QAChain` 编排，全程不可变。组件：`Router`(首版恒文档) → `Retriever`(M2) → `RetrievalGate`(闸①) → `ContextBuilder`(`[n]` 引用 + `Citation` 表) → `PromptBuilder`(防幻觉系统提示 ②) → `LLMClient`(DeepSeek) → `CitationValidator`(闸③)。`QAResult` 聚合答案 + 引用 + 置信度。

**Tech Stack:** 复用 M0-M1/M2 全部依赖；`openai` SDK 调 DeepSeek（OpenAI 兼容，`base_url=https://api.deepseek.com`，`model=deepseek-chat`）。无新增第三方包。

**防幻觉三道闸（对齐 spec §7.2 与 fin-online `anti-hallucination.mdc`）：**

| 闸 | 机制 | 实现组件 | 首版策略 |
|----|------|---------|---------|
| ① 检索门控 | 召回不足直接拒答，不交给 LLM 编 | `RetrievalGate` | 召回数为 0 → 拒答（无阈值歧义）；可选 `min_score` 软阈值，默认不启用 |
| ② 生成约束 | 系统提示硬约束：只基于上下文、`[n]` 引用、无依据说不知道、不编造产品号 | `PromptBuilder` | 硬编码 spec §7.3 骨架 + `temperature=0.1`；可选注入 `anti-hallucination` 规则文本 |
| ③ 事后校验 | 答案引用 `[n]` 必须真实存在于检索结果；答案产品号须出现在上下文 | `CitationValidator` | 正则提取 `[n]` 越界 → 标记 `低置信`；产品号不在上下文 → 标记可疑 |

**接口约定（沿用前序阶段）：**
- M2：`Retriever.search(query, domains=None, top_k=None) -> list[RankedChunk]`（`RankedChunk.citation_label()`、`.score`、`.text`、`.parent_text`、`.parent_node_id`）
- M0-M1：`Settings`（含 `deepseek_api_key/base_url/model`）、`OpenAIEmbedder`
- 本阶段新增：`QAResult`、`Citation`、`GateResult`、`CitationCheck`

---

## Task 1: LLMClient（DeepSeek，OpenAI 兼容，可 mock）

**Files:**
- Create: `src/fin_rag/qa/__init__.py`
- Create: `src/fin_rag/qa/llm_client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: 写失败测试 `tests/test_llm_client.py`**

```python
from unittest.mock import MagicMock, patch

from fin_rag.config import Settings
from fin_rag.qa.llm_client import LLMClient


def _settings():
    return Settings(zhipu_api_key="x", deepseek_api_key="sk-ds",
                    deepseek_base_url="https://api.deepseek.com",
                    deepseek_model="deepseek-chat")


def test_chat_builds_messages_with_system_history_user():
    client = LLMClient(_settings())
    fake = MagicMock()
    fake.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="答案"))]
    )
    with patch.object(client, "_client", fake):
        out = client.chat(
            system="你是助手",
            history=[{"role": "user", "content": "前问"},
                     {"role": "assistant", "content": "前答"}],
            user="这次问题",
        )

    assert out == "答案"
    _, kw = fake.chat.completions.create.call_args
    msgs = kw["messages"]
    assert msgs[0] == {"role": "system", "content": "你是助手"}
    assert msgs[-1] == {"role": "user", "content": "这次问题"}
    assert len(msgs) == 4  # system + 2 history + 1 user
    assert kw["model"] == "deepseek-chat"
    assert kw["temperature"] <= 0.2  # 低温防幻觉


def test_chat_empty_history():
    client = LLMClient(_settings())
    fake = MagicMock()
    fake.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    with patch.object(client, "_client", fake):
        client.chat(system="s", history=[], user="u")
    _, kw = fake.chat.completions.create.call_args
    assert len(kw["messages"]) == 2  # system + user
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/qa/__init__.py`（空）**

```python
```

- [ ] **Step 4: 写 `src/fin_rag/qa/llm_client.py`**

```python
"""DeepSeek LLM 客户端（OpenAI 兼容），低温防幻觉，支持多轮历史。"""

from openai import OpenAI

from fin_rag.config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.deepseek_model
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    def chat(self, system: str, history: list[dict], user: str) -> str:
        """非流式对话；temperature 低以抑制幻觉（M4 前端再包流式）。"""
        messages = [{"role": "system", "content": system}] + list(history) + [
            {"role": "user", "content": user}
        ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_llm_client.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: Commit**

```bash
git add src/fin_rag/qa/__init__.py src/fin_rag/qa/llm_client.py tests/test_llm_client.py
git commit -m "feat: LLMClient（DeepSeek OpenAI 兼容，低温防幻觉，多轮历史）"
```

---

## Task 2: PromptBuilder（防幻觉系统提示硬约束 ②）

**Files:**
- Create: `src/fin_rag/qa/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: 写失败测试 `tests/test_prompts.py`**

```python
from fin_rag.qa.prompts import PromptBuilder


def test_system_prompt_contains_anti_hallucination_rules():
    s = PromptBuilder().build_system(domain="金融")
    # 防幻觉硬约束关键词
    assert "只基于" in s and "检索上下文" in s
    assert "[n]" in s  # 引用标注要求
    assert "知识库未覆盖" in s  # 无依据时明确说不知道
    assert "产品号" in s  # 禁编造具体值
    assert "金融" in s  # domain 注入


def test_system_prompt_appends_extra_rules():
    s = PromptBuilder().build_system(extra_rules="禁止四舍五入金额。")
    assert "禁止四舍五入金额。" in s


def test_user_query_wraps_context_block():
    body = PromptBuilder().build_user(query="账期怎么切", context="[1] 原文")
    assert "<检索上下文>" in body and "</检索上下文>" in body
    assert "[1] 原文" in body
    assert "账期怎么切" in body
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/qa/prompts.py`**

```python
"""系统提示与用户消息组装：防幻觉硬约束（spec §7.3）。"""

_SYSTEM_TEMPLATE = """你是 {domain} 业务知识助手。严格遵循：
1. 只基于<检索上下文>作答，不使用训练记忆，不臆测。
2. 每条结论后用 [n] 标注引用，对应检索上下文中的来源编号；无依据的问题明确回答"知识库未覆盖"，绝不编造产品号/PID/渠道/账期/逻辑。
3. 涉及配置/SQL/接口时，给出文件路径与章节，必要时贴原文片段。
4. 业务术语用文档原词；金额/比例/产品号原样引用，不做四舍五入或推断。
5. 若多份文档冲突，列出各来源与差异，交由用户判断，不擅自裁决。
{extra_rules}"""


class PromptBuilder:
    def build_system(self, domain: str = "业务", extra_rules: str = "") -> str:
        tail = f"\n额外约束：\n{extra_rules}" if extra_rules else ""
        return _SYSTEM_TEMPLATE.format(domain=domain, extra_rules=tail)

    def build_user(self, query: str, context: str) -> str:
        return f"<检索上下文>\n{context}\n</检索上下文>\n\n问题：{query}"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/qa/prompts.py tests/test_prompts.py
git commit -m "feat: PromptBuilder（防幻觉系统提示硬约束 + 上下文包裹）"
```

---

## Task 3: ContextBuilder + Citation（`[n]` 引用编号 + 溯源表）

**Files:**
- Create: `src/fin_rag/qa/context.py`
- Test: `tests/test_context_builder.py`

- [ ] **Step 1: 写失败测试 `tests/test_context_builder.py`**

```python
from fin_rag.qa.context import ContextBuilder, Citation
from fin_rag.retrieval.types import RankedChunk


def _chunk(text, **over):
    base = dict(
        node_id="n", text=text, score=0.8, project="fin-online", domain="fin-online",
        source_type="doc", doc_type="tech_plan", product_code=None, channel=None,
        heading_path="账期", file_path="docs/a.md", parent_node_id=None,
    )
    base.update(over)
    return RankedChunk(**base)


def test_build_numbers_chunks_and_returns_citations():
    ctx, cites = ContextBuilder().build([_chunk("正文1"), _chunk("正文2")])
    assert "[1]" in ctx and "[2]" in ctx
    assert "正文1" in ctx and "正文2" in ctx
    assert len(cites) == 2
    assert isinstance(cites[0], Citation)
    assert cites[0].id == 1
    assert cites[1].id == 2
    assert cites[0].file_path == "docs/a.md"
    assert "账期" in cites[0].heading_path


def test_build_empty_chunks_yields_empty():
    ctx, cites = ContextBuilder().build([])
    assert ctx == ""
    assert cites == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_context_builder.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/qa/context.py`**

```python
"""上下文组装：把 RankedChunk 列表转为带 [n] 引用编号的文本 + 溯源 Citation 表。"""

from dataclasses import dataclass

from fin_rag.retrieval.types import RankedChunk


@dataclass(frozen=True)
class Citation:
    id: int                 # 引用编号（与上下文 [n] 对应）
    label: str              # 溯源标签 [domain | 文件 | §章节]
    file_path: str
    heading_path: str
    text: str


class ContextBuilder:
    def build(self, chunks: list[RankedChunk]) -> tuple[str, list[Citation]]:
        lines: list[str] = []
        citations: list[Citation] = []
        for i, c in enumerate(chunks, start=1):
            lines.append(f"[{i}] {c.citation_label()} {c.text}")
            citations.append(
                Citation(
                    id=i,
                    label=c.citation_label(),
                    file_path=c.file_path,
                    heading_path=c.heading_path,
                    text=c.text,
                )
            )
        return "\n\n".join(lines), citations
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_context_builder.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/qa/context.py tests/test_context_builder.py
git commit -m "feat: ContextBuilder + Citation（[n] 引用编号 + 溯源表）"
```

---

## Task 4: RetrievalGate（防幻觉闸①：召回门控）

**Files:**
- Create: `src/fin_rag/qa/guardrails.py`
- Test: `tests/test_retrieval_gate.py`

> 说明：`guardrails.py` 同时承载闸①与闸③（`CitationValidator`，Task 5）。首版门控只用「召回数为 0 → 拒答」（无 RRF 分数阈值歧义）；`min_score` 软阈值默认 0.0（不启用），留待 M4 评测调优。

- [ ] **Step 1: 写失败测试 `tests/test_retrieval_gate.py`**

```python
from fin_rag.qa.guardrails import RetrievalGate, GateResult
from fin_rag.retrieval.types import RankedChunk


def _chunk(score):
    return RankedChunk(
        node_id="n", text="t", score=score, project="fin-online", domain="fin-online",
        source_type="doc", doc_type="tech_plan", product_code=None, channel=None,
        heading_path="h", file_path="docs/a.md", parent_node_id=None,
    )


def test_empty_recall_is_gated():
    res = RetrievalGate().decide([])
    assert isinstance(res, GateResult)
    assert res.passed is False
    assert "召回" in res.reason


def test_non_empty_passes_by_default():
    res = RetrievalGate().decide([_chunk(0.05)])
    assert res.passed is True


def test_low_score_gated_when_min_score_enabled():
    gate = RetrievalGate(min_score=0.1)
    res = gate.decide([_chunk(0.05)])
    assert res.passed is False
    assert "置信度" in res.reason


def test_high_score_passes_when_min_score_enabled():
    gate = RetrievalGate(min_score=0.1)
    res = gate.decide([_chunk(0.5)])
    assert res.passed is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_retrieval_gate.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/qa/guardrails.py`（先只放闸①；闸③ Task 5 追加）**

```python
"""防幻觉三道闸：检索门控(①) + 引用事后校验(③)。闸② 见 prompts.py 系统提示。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str


class RetrievalGate:
    """闸①：召回不足直接拒答，不交给 LLM 编。"""

    def __init__(self, min_score: float = 0.0) -> None:
        self.min_score = min_score

    def decide(self, chunks: list) -> GateResult:
        if not chunks:
            return GateResult(False, "无召回：知识库未覆盖该问题")
        if self.min_score > 0 and max(c.score for c in chunks) < self.min_score:
            return GateResult(False, "召回置信度不足")
        return GateResult(True, "")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_retrieval_gate.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/qa/guardrails.py tests/test_retrieval_gate.py
git commit -m "feat: RetrievalGate（防幻觉闸① 召回门控拒答）"
```

---

## Task 5: CitationValidator（防幻觉闸③：引用事后校验）

**Files:**
- Modify: `src/fin_rag/qa/guardrails.py`（追加闸③）
- Test: `tests/test_citation_validator.py`

- [ ] **Step 1: 写失败测试 `tests/test_citation_validator.py`**

```python
from fin_rag.qa.context import Citation
from fin_rag.qa.guardrails import CitationValidator, CitationCheck


def _cite(i, text=""):
    return Citation(id=i, label="[d|f|§h]", file_path="f", heading_path="h", text=text)


def test_valid_answer_passes():
    cites = [_cite(1, "产品号 300007"), _cite(2, "账期")]
    check = CitationValidator().validate("切换逻辑见 [1]。", cites)
    assert isinstance(check, CitationCheck)
    assert check.valid is True
    assert check.invalid_refs == []
    assert check.suspect_products == []


def test_out_of_range_ref_marked_invalid():
    cites = [_cite(1)]
    check = CitationValidator().validate("见 [9]。", cites)  # 只有 [1]，[9] 越界
    assert check.valid is False
    assert 9 in check.invalid_refs


def test_product_not_in_context_marked_suspect():
    cites = [_cite(1, "产品号 300007")]
    check = CitationValidator().validate("产品 300008 见 [1]。", cites)  # 300008 上下文没有
    assert check.valid is False
    assert "300008" in check.suspect_products
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_citation_validator.py -v`
Expected: FAIL（`ImportError: cannot import name 'CitationValidator'`）

- [ ] **Step 3: 在 `src/fin_rag/qa/guardrails.py` 末尾追加闸③**

在文件末尾追加（保留已有的 `GateResult` / `RetrievalGate`）：

```python
import re
from fin_rag.qa.context import Citation

_CITE_RE = re.compile(r"\[(\d+)\]")
_PRODUCT_RE = re.compile(r"\b(3\d{5}|6\d{5})\b")


@dataclass(frozen=True)
class CitationCheck:
    valid: bool
    invalid_refs: list  # 答案里越界/不存在的 [n]
    suspect_products: list  # 答案里上下文未出现的产品号


class CitationValidator:
    """闸③：核对答案引用是否真实存在于检索结果；产品号须出现在上下文。"""

    def validate(self, answer: str, citations: list[Citation]) -> CitationCheck:
        max_id = len(citations)
        cited = {int(n) for n in _CITE_RE.findall(answer)}
        invalid_refs = sorted(n for n in cited if n < 1 or n > max_id)

        context_text = " ".join(c.text for c in citations)
        ctx_products = set(_PRODUCT_RE.findall(context_text))
        ans_products = set(_PRODUCT_RE.findall(answer))
        suspect_products = sorted(p for p in ans_products if p not in ctx_products)

        valid = not invalid_refs and not suspect_products
        return CitationCheck(valid=valid, invalid_refs=invalid_refs, suspect_products=suspect_products)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_citation_validator.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/qa/guardrails.py tests/test_citation_validator.py
git commit -m "feat: CitationValidator（防幻觉闸③ 引用越界/产品号事后校验）"
```

---

## Task 6: Router（首版恒走文档，DataQueryProvider 接口预留）

**Files:**
- Create: `src/fin_rag/qa/router.py`
- Test: `tests/test_router.py`

- [ ] **Step 1: 写失败测试 `tests/test_router.py`**

```python
import pytest

from fin_rag.qa.router import Router


def test_router_defaults_to_document():
    assert Router().route("任意问题") == "document"


def test_data_query_provider_protocol_exists():
    # DataQueryProvider 接口预留：无 provider 时，数据类问题也走文档分支
    r = Router(data_provider=None)
    assert r.route("统计某产品数量") == "document"


def test_router_with_provider_dispatches_data(monkeypatch):
    provider = type("P", (), {"is_data_query": staticmethod(lambda q: True)})()
    r = Router(data_provider=provider)
    assert r.route("数量统计") == "data"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_router.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 `src/fin_rag/qa/router.py`**

```python
"""文档/数据路由：首版恒走文档分支；DataQueryProvider 接口预留（二期 NL2SQL 插入）。"""

from typing import Optional, Protocol


class DataQueryProvider(Protocol):
    """二期实现：is_data_query 判定 + 提供 NL2SQL 能力。首版不实现。"""

    def is_data_query(self, query: str) -> bool: ...


class Router:
    def __init__(self, data_provider: Optional[DataQueryProvider] = None) -> None:
        self.data_provider = data_provider

    def route(self, query: str) -> str:
        if self.data_provider is not None and self.data_provider.is_data_query(query):
            return "data"
        return "document"  # 首版默认
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_router.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/qa/router.py tests/test_router.py
git commit -m "feat: Router（文档/数据路由，首版恒文档，DataQueryProvider 预留）"
```

---

## Task 7: QAChain 编排 + QAResult（端到端：门控→组装→生成→校验）

**Files:**
- Create: `src/fin_rag/qa/chain.py`
- Test: `tests/test_qa_chain.py`

- [ ] **Step 1: 写失败测试 `tests/test_qa_chain.py`（全组件 mock，验证编排与三道闸）**

```python
from unittest.mock import MagicMock

from fin_rag.qa.chain import QAChain, GATE_REFUSAL_TEMPLATE
from fin_rag.qa.context import Citation
from fin_rag.qa.guardrails import CitationCheck, GateResult
from fin_rag.retrieval.types import RankedChunk


def _chunk():
    return RankedChunk(
        node_id="n", text="账期切换逻辑见 300007", score=0.8, project="fin-online",
        domain="fin-online", source_type="doc", doc_type="business_logic",
        product_code="300007", channel=None, heading_path="账期",
        file_path="docs/a.md", parent_node_id=None,
    )


def _chain(retriever, gate_passed=True, validator_valid=True):
    retriever.search.return_value = [_chunk()] if gate_passed else []
    llm = MagicMock()
    llm.chat.return_value = "账期见 [1]，产品 300007。"
    prompt = MagicMock()
    prompt.build_system.return_value = "SYS"
    prompt.build_user.return_value = "USER"
    ctx = MagicMock()
    ctx.build.return_value = ("[1] 账期切换逻辑见 300007", [Citation(1, "[d|f|§h]", "f", "h", "账期 300007")])
    gate = MagicMock()
    gate.decide.return_value = GateResult(gate_passed, "" if gate_passed else "无召回")
    validator = MagicMock()
    validator.validate.return_value = CitationCheck(valid=validator_valid, invalid_refs=[], suspect_products=[])
    router = MagicMock()
    router.route.return_value = "document"
    return QAChain(retriever, llm, prompt, ctx, gate, validator, router)


def test_ask_full_pipeline_returns_answer_with_citations():
    retriever = MagicMock()
    chain = _chain(retriever)
    res = chain.ask("账期怎么切")

    assert res.answer == "账期见 [1]，产品 300007。"
    assert res.gated is False
    assert res.confidence == "high"
    assert len(res.citations) == 1
    retriever.search.assert_called_once()
    chain._llm.chat.assert_called_once()


def test_ask_refuses_when_gate_blocks():
    retriever = MagicMock()
    chain = _chain(retriever, gate_passed=False)
    res = chain.ask("无关问题")

    assert res.gated is True
    assert res.answer == GATE_REFUSAL_TEMPLATE
    assert res.confidence == "low"
    chain._llm.chat.assert_not_called()  # 门控拦截，不调 LLM


def test_ask_marks_low_confidence_when_citation_invalid():
    retriever = MagicMock()
    chain = _chain(retriever, gate_passed=True, validator_valid=False)
    res = chain.ask("账期")

    assert res.citation_check.valid is False
    assert res.confidence == "low"


def test_ask_passes_domains_to_retriever():
    retriever = MagicMock()
    chain = _chain(retriever)
    chain.ask("账期", domains=["fin-online"])
    _, kw = retriever.search.call_args
    assert kw["domains"] == ["fin-online"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_qa_chain.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'fin_rag.qa.chain'`）

- [ ] **Step 3: 写 `src/fin_rag/qa/chain.py`**

```python
"""QAChain：防幻觉问答编排主流程（路由→检索→门控①→组装→系统提示②→生成→校验③）。"""

from dataclasses import dataclass
from typing import Optional

from fin_rag.qa.context import Citation, ContextBuilder
from fin_rag.qa.guardrails import CitationCheck, CitationValidator, RetrievalGate
from fin_rag.qa.llm_client import LLMClient
from fin_rag.qa.prompts import PromptBuilder
from fin_rag.qa.router import Router
from fin_rag.retrieval.retriever import Retriever

GATE_REFUSAL_TEMPLATE = "知识库未覆盖该问题，建议补充文档或换个问法。"


@dataclass(frozen=True)
class QAResult:
    answer: str
    citations: list[Citation]
    gated: bool
    gate_reason: str
    citation_check: Optional[CitationCheck]
    confidence: str  # high | low


class QAChain:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        prompt_builder: PromptBuilder,
        context_builder: ContextBuilder,
        gate: RetrievalGate,
        validator: CitationValidator,
        router: Router,
        domain: str = "金融",
        extra_rules: str = "",
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt = prompt_builder
        self._context = context_builder
        self._gate = gate
        self._validator = validator
        self._router = router
        self._domain = domain
        self._extra_rules = extra_rules

    def ask(
        self,
        query: str,
        history: Optional[list[dict]] = None,
        domains: Optional[list[str]] = None,
    ) -> QAResult:
        route = self._router.route(query)
        if route != "document":
            # 二期前：数据类问题明确提示
            return QAResult(
                answer="数据查询二期支持，当前仅文档问答。",
                citations=[], gated=False, gate_reason="data_route",
                citation_check=None, confidence="low",
            )

        chunks = self._retriever.search(query, domains=domains)

        gate_result = self._gate.decide(chunks)
        if not gate_result.passed:
            return QAResult(
                answer=GATE_REFUSAL_TEMPLATE,
                citations=[], gated=True, gate_reason=gate_result.reason,
                citation_check=None, confidence="low",
            )

        context, citations = self._context.build(chunks)
        system = self._prompt.build_system(domain=self._domain, extra_rules=self._extra_rules)
        user = self._prompt.build_user(query=query, context=context)
        answer = self._llm.chat(system=system, history=history or [], user=user)

        check = self._validator.validate(answer, citations)
        confidence = "high" if check.valid else "low"
        return QAResult(
            answer=answer,
            citations=citations,
            gated=False,
            gate_reason="",
            citation_check=check,
            confidence=confidence,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_qa_chain.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add src/fin_rag/qa/chain.py tests/test_qa_chain.py
git commit -m "feat: QAChain + QAResult（防幻觉三道闸端到端编排 + 置信度标记）"
```

---

## Task 8: CLI `fin-rag ask` + 端到端验证

**Files:**
- Modify: `src/fin_rag/cli.py`（新增 `ask` 子命令 + QA 工厂）
- Test: `tests/test_cli_ask.py`

- [ ] **Step 1: 写失败测试 `tests/test_cli_ask.py`（mock QA 工厂，验证 CLI 透传与输出）**

```python
from unittest.mock import MagicMock, patch

from fin_rag.cli import main


def test_cli_ask_prints_answer_and_citations(capsys):
    fake_result = MagicMock()
    fake_result.answer = "账期切换见 [1]。"
    fake_result.confidence = "high"
    fake_result.gated = False
    fake_result.gate_reason = ""
    fake_result.citations = [MagicMock(id=1, label="[fin-online|docs/a.md|§账期]")]
    with patch("fin_rag.cli._build_qa_chain") as mk:
        mk.return_value.ask.return_value = fake_result
        code = main(["ask", "账期怎么切"])
    out = capsys.readouterr().out
    assert code == 0
    assert "账期切换见 [1]。" in out
    assert "fin-online|docs/a.md" in out


def test_cli_ask_passes_domain_flag(capsys):
    fake_result = MagicMock()
    fake_result.answer = "ok"; fake_result.confidence = "high"
    fake_result.gated = False; fake_result.gate_reason = ""
    fake_result.citations = []
    with patch("fin_rag.cli._build_qa_chain") as mk:
        mk.return_value.ask.return_value = fake_result
        main(["ask", "账期", "--domain", "opdata"])
    _, kw = mk.return_value.ask.call_args
    assert kw["domains"] == ["opdata"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cli_ask.py -v`
Expected: FAIL（`AttributeError: ... '_build_qa_chain'` 或 subparser 缺失）

- [ ] **Step 3: 改 `src/fin_rag/cli.py`（新增 QA 工厂 + ask 子命令）**

在 import 区追加（`load_settings`/`OpenAIEmbedder` 等已在 M0-M1/M2 import，不重复）：

```python
from fin_rag.qa.chain import QAChain
from fin_rag.qa.context import ContextBuilder
from fin_rag.qa.guardrails import CitationValidator, RetrievalGate
from fin_rag.qa.llm_client import LLMClient
from fin_rag.qa.prompts import PromptBuilder
from fin_rag.qa.router import Router
```

在 `_build_retriever()`（M2 新增）之后追加 QA 工厂（**复用 `_build_retriever()`**，避免重复构造检索栈）：

```python
def _build_qa_chain():
    settings = load_settings(CONFIG_DIR / "settings.yaml")
    retriever = _build_retriever()
    return QAChain(
        retriever=retriever,
        llm=LLMClient(settings),
        prompt_builder=PromptBuilder(),
        context_builder=ContextBuilder(),
        gate=RetrievalGate(),
        validator=CitationValidator(),
        router=Router(),
    )
```

在 `main()` 的 subparser 注册区（`search` 之后）追加：

```python
    ask_p = sub.add_parser("ask", help="防幻觉问答")
    ask_p.add_argument("query", help="问题")
    ask_p.add_argument("--domain", action="append", default=None, help="按 domain 筛选（可多次）")
```

在 `if args.cmd == "search":` 分支之后追加 `ask` 分支：

```python
    if args.cmd == "ask":
        try:
            res = _build_qa_chain().ask(args.query, domains=args.domain)
            print(res.answer)
            for c in res.citations:
                print(f"  [{c.id}] {c.label}")
            print(f"[置信度: {res.confidence}" + (f" · 拒答: {res.gate_reason}]" if res.gated else "]"))
            return 0
        except Exception as exc:
            log.error("问答失败：%s", exc)
            return 1
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cli_ask.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 运行全量测试 + 覆盖率**

Run: `pytest -v`
Expected: 全部 PASS，`fin_rag` 覆盖率 ≥ 80%

- [ ] **Step 6: 端到端手动验证（需 Qdrant + 真实 API key + 已建索引）**

```bash
cd /Users/wangfei/yulore/fin-rag
docker compose up -d qdrant
source .venv/bin/activate
fin-rag index          # 确保 M2 双向量索引就绪
# 防幻觉正例：库里有的业务逻辑
fin-rag ask "账期切换逻辑是什么"
# 防幻觉闸①：库里没有的内容 → 应拒答
fin-rag ask "火星殖民的金融政策"
# 引用溯源：答案末尾应列出 [n] 来源标签
fin-rag ask "日志三级备份怎么做" --domain fin-online
```
Expected：
- 正例输出答案 + 引用标签 `[n] [domain|文件|§章节]`，置信度 `high`。
- 无关问题命中闸① → 输出「知识库未覆盖该问题…」，置信度 `low`，未调用 DeepSeek。
- 若 LLM 编造越界引用 `[n]` 或上下文外产品号，置信度降为 `low`（闸③ 拦截）。

- [ ] **Step 7: Commit**

```bash
git add src/fin_rag/cli.py tests/test_cli_ask.py
git commit -m "feat: CLI ask 防幻觉问答（门控→检索→生成→引用校验，端到端可验证）"
```

---

## 完成标准（M3）

- [ ] 全部 8 个任务测试通过，`pytest` 覆盖率 ≥ 80%
- [ ] `fin-rag ask` 端到端出带引用的答案；防幻觉三道闸各有单测覆盖
- [ ] 闸①：无召回 → 拒答、不调 LLM
- [ ] 闸③：引用越界 / 产品号不在上下文 → 置信度降为 `low`
- [ ] `QAResult` 含 `answer/citations/confidence/gated`，可被 M4 前端直接消费

## 下一步

M3 完成后，进入**第四份计划（M4-M5）**：Gradio `ChatInterface` 前端（多轮 + 流式 + 溯源面板 + 反馈）→ Golden Set 评测（召回命中率 / 引用准确率 / 拒答准确率 / 忠实度）→ 增量同步（hash+mtime 幂等）+ Docker 部署（团队内网 URL）。
