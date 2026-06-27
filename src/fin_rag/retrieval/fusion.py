from __future__ import annotations

from fin_rag.config import Settings
from fin_rag.retrieval.filters import normalize_domains
from fin_rag.retrieval.query_rewrite import normalize_query
from fin_rag.store.runtime import build_docstore, build_embedding, build_qdrant_vector_store


def build_query_bundle(query: str) -> str:
    return normalize_query(query).strip()


def build_vector_retriever(settings: Settings, domains: list[str] | None = None, index=None):
    vector_index = index or build_vector_index(settings)
    filters = None
    normalized_domains = normalize_domains(domains)
    if normalized_domains:
        from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="domain", value=normalized_domains, operator=FilterOperator.IN),
            ]
        )
    try:
        from llama_index.core.retrievers import VectorIndexRetriever
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index retrieval dependencies are not installed."
        ) from exc
    return VectorIndexRetriever(index=vector_index, similarity_top_k=settings.fusion_top_k, filters=filters)


def build_vector_index(settings: Settings):
    try:
        from llama_index.core.indices.vector_store import VectorStoreIndex
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index retrieval dependencies are not installed."
        ) from exc

    embed_model = build_embedding(settings)
    vector_store = build_qdrant_vector_store(settings)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)


def build_retriever(
    settings: Settings,
    domains: list[str] | None = None,
    llm=None,
    index=None,
):
    try:
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index retrieval dependencies are not installed."
        ) from exc

    vector_index = index or build_vector_index(settings)
    vector_retriever = build_vector_retriever(settings=settings, domains=domains, index=vector_index)
    docstore = build_docstore(settings)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=settings.fusion_top_k,
    )
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        llm=llm,
        similarity_top_k=settings.fusion_top_k,
        num_queries=1,
        mode="reciprocal_rerank",
    )
