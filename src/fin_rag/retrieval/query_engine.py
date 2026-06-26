from __future__ import annotations

from fin_rag.config import Settings
from fin_rag.llm import build_llm
from fin_rag.retrieval.fusion import build_retriever, build_vector_index


def build_query_engine(
    settings: Settings,
    domains: list[str] | None = None,
    llm=None,
    node_postprocessors=None,
):
    try:
        from llama_index.core.query_engine import CitationQueryEngine
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index query engine dependencies are not installed."
        ) from exc

    active_llm = llm or build_llm(settings)
    index = build_vector_index(settings)
    retriever = build_retriever(settings, domains=domains, llm=active_llm, index=index)
    return CitationQueryEngine.from_args(
        index=index,
        llm=active_llm,
        retriever=retriever,
        node_postprocessors=node_postprocessors,
        similarity_top_k=settings.fusion_top_k,
        citation_chunk_size=512,
    )
