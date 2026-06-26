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


def ask_question(settings: Settings, question: str, domains: list[str] | None = None, llm=None) -> QAResult:
    try:
        from llama_index.core.indices.postprocessor import SimilarityPostprocessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index postprocessor dependencies are not installed."
        ) from exc

    active_llm = llm or build_llm(settings)
    query_engine = build_query_engine(
        settings,
        domains=domains,
        llm=active_llm,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=settings.similarity_cutoff)],
    )
    response = query_engine.query(question)
    source_contexts = _build_citation_contexts(response.source_nodes)
    answer = (response.response or "").strip()
    validation = validate_citations(answer, source_contexts)

    refused = not source_contexts or not validation["valid"] or not answer
    if refused:
        answer = REFUSAL_TEMPLATE

    return QAResult(
        answer=answer,
        sources=[
            {
                "score": node.score,
                "text": node.node.text[:400],
                "metadata": node.node.metadata,
            }
            for node in response.source_nodes
        ],
        refused=refused,
        validation=validation,
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
