from __future__ import annotations

import inspect
from pathlib import Path

from t_rag.config import Settings


def build_qdrant_vector_store(settings: Settings):
    try:
        from qdrant_client import AsyncQdrantClient, QdrantClient
        from llama_index.vector_stores.qdrant import QdrantVectorStore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant-client or llama-index vector store integration is not installed."
        ) from exc

    _patch_qdrant_client_compatibility(QdrantClient)
    _patch_qdrant_client_compatibility(AsyncQdrantClient)
    client = QdrantClient(url=settings.qdrant_url)
    aclient = AsyncQdrantClient(url=settings.qdrant_url)
    return QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=settings.qdrant_collection,
    )


def _patch_qdrant_client_compatibility(qdrant_client_class) -> None:
    if hasattr(qdrant_client_class, "search") and hasattr(qdrant_client_class, "search_batch"):
        return

    is_async_client = inspect.iscoroutinefunction(getattr(qdrant_client_class, "query_points", None))

    if not hasattr(qdrant_client_class, "search"):

        async def async_search(
            self, collection_name, query_vector=None, query_filter=None, limit=10, **kwargs
        ):
            query = query_vector if query_vector is not None else kwargs.pop("query", None)
            response = await self.query_points(
                collection_name=collection_name,
                query=query,
                query_filter=query_filter,
                limit=limit,
                with_payload=kwargs.pop("with_payload", True),
                with_vectors=kwargs.pop("with_vectors", False),
                **kwargs,
            )
            return response.points

        def search(self, collection_name, query_vector=None, query_filter=None, limit=10, **kwargs):
            query = query_vector if query_vector is not None else kwargs.pop("query", None)
            response = self.query_points(
                collection_name=collection_name,
                query=query,
                query_filter=query_filter,
                limit=limit,
                with_payload=kwargs.pop("with_payload", True),
                with_vectors=kwargs.pop("with_vectors", False),
                **kwargs,
            )
            return response.points

        if is_async_client:
            qdrant_client_class.search = async_search
        else:
            qdrant_client_class.search = search

    if not hasattr(qdrant_client_class, "search_batch"):

        async def async_search_batch(self, collection_name, requests, **kwargs):
            responses = []
            for request in requests:
                vector = getattr(request, "vector", None)
                query = getattr(vector, "vector", vector)
                response = await self.query_points(
                    collection_name=collection_name,
                    query=query,
                    query_filter=getattr(request, "filter", None),
                    limit=getattr(request, "limit", 10),
                    with_payload=getattr(request, "with_payload", True),
                    with_vectors=getattr(request, "with_vector", False),
                    **kwargs,
                )
                responses.append(response.points)
            return responses

        def search_batch(self, collection_name, requests, **kwargs):
            responses = []
            for request in requests:
                vector = getattr(request, "vector", None)
                query = getattr(vector, "vector", vector)
                response = self.query_points(
                    collection_name=collection_name,
                    query=query,
                    query_filter=getattr(request, "filter", None),
                    limit=getattr(request, "limit", 10),
                    with_payload=getattr(request, "with_payload", True),
                    with_vectors=getattr(request, "with_vector", False),
                    **kwargs,
                )
                responses.append(response.points)
            return responses

        if is_async_client:
            qdrant_client_class.search_batch = async_search_batch
        else:
            qdrant_client_class.search_batch = search_batch


def build_docstore(settings: Settings | None = None):
    try:
        from llama_index.core.storage.docstore import SimpleDocumentStore
    except ModuleNotFoundError:
        try:
            from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Unable to import LlamaIndex SimpleDocumentStore.") from exc
    if settings is not None:
        persist_path = Path(settings.docstore_path) / "docstore.json"
        if persist_path.exists():
            return SimpleDocumentStore.from_persist_path(str(persist_path))
    return SimpleDocumentStore()


def build_storage_context(settings: Settings, vector_store=None, docstore=None):
    try:
        from llama_index.core import StorageContext
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index is not installed. Install project dependencies before building storage context."
        ) from exc

    return StorageContext.from_defaults(
        vector_store=vector_store or build_qdrant_vector_store(settings),
        docstore=docstore or build_docstore(settings),
        )


def persist_docstore(docstore, settings: Settings) -> None:
    persist_dir = settings.docstore_path
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(docstore, "persist"):
        docstore.persist(str(Path(persist_dir) / "docstore.json"))
