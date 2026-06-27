from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from fin_rag.config import Settings
from fin_rag.store.runtime import build_docstore, build_embedding, build_qdrant_vector_store, persist_docstore

_settings_lock = threading.Lock()


@dataclass(slots=True)
class IndexExecutionResult:
    document_count: int
    node_count: int
    collection_name: str
    qdrant_url: str


def run_llama_index(settings: Settings, documents: list[Any]) -> IndexExecutionResult:
    IngestionPipeline, HierarchicalNodeParser = _load_ingestion_types()
    embed_model = build_embedding(settings)
    vector_store = _build_qdrant_vector_store(settings)
    docstore = build_docstore()

    pipeline = IngestionPipeline(
        transformations=[
            HierarchicalNodeParser.from_defaults(chunk_sizes=settings.chunk_sizes),
            embed_model,
        ],
        vector_store=vector_store,
        docstore=docstore,
    )
    nodes = pipeline.run(documents=documents, show_progress=True)
    persist_docstore(docstore=docstore, settings=settings)
    return IndexExecutionResult(
        document_count=len(documents),
        node_count=len(nodes),
        collection_name=settings.qdrant_collection,
        qdrant_url=settings.qdrant_url,
    )


def _load_ingestion_types():
    try:
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import HierarchicalNodeParser
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index is not installed. Install project dependencies before running real indexing."
        ) from exc
    return IngestionPipeline, HierarchicalNodeParser


def _build_qdrant_vector_store(settings: Settings):
    try:
        from llama_index.core import Settings as LlamaSettings
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index is not installed. Install project dependencies before running real indexing."
        ) from exc

    with _settings_lock:
        LlamaSettings.embed_model = build_embedding(settings)
    return build_qdrant_vector_store(settings)
