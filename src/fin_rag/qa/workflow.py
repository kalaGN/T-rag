from __future__ import annotations

from dataclasses import dataclass

from fin_rag.config import Settings
from fin_rag.llm import build_llm
from fin_rag.qa.citation import CitationContext, validate_citations
from fin_rag.qa.prompts import REFUSAL_TEMPLATE
from fin_rag.retrieval.query_engine import build_query_engine


@dataclass(slots=True)
class QAResult:
    answer: str
    sources: list[dict[str, object]]
    refused: bool
    validation: dict[str, object]
    raw_answer: str


def ask_question(settings: Settings, question: str, domains: list[str] | None = None, llm=None) -> QAResult:
    try:
        from llama_index.core.indices.postprocessor import SimilarityPostprocessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index postprocessor dependencies are not installed."
        ) from exc

    active_llm = llm or build_llm(settings)
    similarity_cutoff = _resolve_similarity_cutoff(question, settings.similarity_cutoff)
    result = _run_query(
        settings=settings,
        question=question,
        domains=domains,
        llm=active_llm,
        similarity_cutoff=similarity_cutoff,
    )
    if result.validation["refusal_reason"] in {"no_sources", "empty_answer"}:
        fallback = _run_query(
            settings=settings,
            question=question,
            domains=domains,
            llm=active_llm,
            similarity_cutoff=None,
        )
        if fallback.source_contexts:
            result = fallback

    refused = not result.source_contexts or not result.validation["valid"] or not result.answer
    if refused:
        answer = REFUSAL_TEMPLATE
    else:
        answer = result.answer

    return QAResult(
        answer=answer,
        sources=[
            {
                "score": node.score,
                "text": node.node.text[:400],
                "metadata": node.node.metadata,
            }
            for node in result.source_nodes
        ],
        refused=refused,
        validation=result.validation,
        raw_answer=result.answer,
    )


def _build_citation_contexts(source_nodes) -> list[CitationContext]:
    contexts: list[CitationContext] = []
    for index, node in enumerate(source_nodes, start=1):
        metadata = node.node.metadata or {}
        contexts.append(
            CitationContext(
                index=index,
                domain=str(metadata.get("domain", "unknown")),
                file_path=str(metadata.get("file_path", metadata.get("source_file", "unknown"))),
                heading_path=str(metadata.get("heading_path", metadata.get("section", "unknown"))),
                text=node.node.text,
            )
        )
    return contexts


def _resolve_similarity_cutoff(question: str, default_cutoff: float) -> float:
    normalized = "".join(question.split())
    if len(normalized) <= 2:
        return min(default_cutoff, 0.15)
    if len(normalized) <= 4:
        return min(default_cutoff, 0.2)
    if len(normalized) <= 8:
        return min(default_cutoff, 0.3)
    return default_cutoff


@dataclass(slots=True)
class _QueryRunResult:
    answer: str
    source_nodes: list[object]
    source_contexts: list[CitationContext]
    validation: dict[str, object]


def _run_query(
    settings: Settings,
    question: str,
    domains: list[str] | None,
    llm,
    similarity_cutoff: float | None,
) -> _QueryRunResult:
    node_postprocessors = []
    if similarity_cutoff is not None:
        from llama_index.core.indices.postprocessor import SimilarityPostprocessor

        node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]

    query_engine = build_query_engine(
        settings,
        domains=domains,
        llm=llm,
        node_postprocessors=node_postprocessors or None,
    )
    response = query_engine.query(question)
    source_contexts = _build_citation_contexts(response.source_nodes)
    answer = (response.response or "").strip()
    validation = validate_citations(answer, source_contexts)
    validation["raw_answer"] = answer
    validation["refusal_reason"] = _resolve_refusal_reason(
        source_contexts=source_contexts,
        answer=answer,
        validation=validation,
    )
    return _QueryRunResult(
        answer=answer,
        source_nodes=response.source_nodes,
        source_contexts=source_contexts,
        validation=validation,
    )


def _resolve_refusal_reason(source_contexts: list[CitationContext], answer: str, validation: dict[str, object]) -> str | None:
    if not source_contexts:
        return "no_sources"
    if not answer:
        return "empty_answer"
    if validation.get("invalid_refs"):
        return "invalid_refs"
    if validation.get("suspect_products"):
        return "suspect_products"
    return None
