import asyncio
import sys
from types import ModuleType

from fin_rag.config import Settings
from fin_rag.store.runtime import _patch_qdrant_client_compatibility
from fin_rag.store.runtime import build_qdrant_vector_store


class _DummyResponse:
    def __init__(self, points):
        self.points = points


def test_patch_qdrant_client_compatibility_adds_search_and_search_batch():
    class DummyClient:
        def __init__(self):
            self.calls = []

        def query_points(self, **kwargs):
            self.calls.append(kwargs)
            return _DummyResponse([kwargs["query"]])

    _patch_qdrant_client_compatibility(DummyClient)
    client = DummyClient()

    assert client.search(collection_name="docs", query_vector=[1.0, 2.0], limit=3) == [[1.0, 2.0]]
    assert client.calls[0]["query"] == [1.0, 2.0]
    assert client.search_batch(
        collection_name="docs",
        requests=[type("Request", (), {"vector": type("Vector", (), {"vector": [3.0]})(), "limit": 1})()],
    ) == [[[3.0]]]


def test_patch_qdrant_client_compatibility_adds_async_methods():
    class DummyAsyncClient:
        def __init__(self):
            self.calls = []

        async def query_points(self, **kwargs):
            self.calls.append(kwargs)
            return _DummyResponse([kwargs["query"]])

    _patch_qdrant_client_compatibility(DummyAsyncClient)
    client = DummyAsyncClient()

    async def run() -> None:
        assert await client.search(collection_name="docs", query_vector=[1.0]) == [[1.0]]
        assert await client.search_batch(
            collection_name="docs",
            requests=[type("Request", (), {"vector": type("Vector", (), {"vector": [2.0]})(), "limit": 1})()],
        ) == [[[2.0]]]

    asyncio.run(run())


def test_build_qdrant_vector_store_passes_async_client(monkeypatch):
    qdrant_module = ModuleType("qdrant_client")

    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    qdrant_module.QdrantClient = DummyClient
    qdrant_module.AsyncQdrantClient = DummyAsyncClient
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_module)

    vector_store_module = ModuleType("llama_index.vector_stores.qdrant")

    class DummyVectorStore:
        last_kwargs = None

        def __init__(self, **kwargs):
            DummyVectorStore.last_kwargs = kwargs

    vector_store_module.QdrantVectorStore = DummyVectorStore
    monkeypatch.setitem(sys.modules, "llama_index.vector_stores.qdrant", vector_store_module)

    settings = Settings(qdrant_url="http://example.test:6333", qdrant_collection="docs")
    store = build_qdrant_vector_store(settings)

    assert isinstance(store, DummyVectorStore)
    assert DummyVectorStore.last_kwargs["client"].kwargs["url"] == "http://example.test:6333"
    assert DummyVectorStore.last_kwargs["aclient"].kwargs["url"] == "http://example.test:6333"
    assert DummyVectorStore.last_kwargs["collection_name"] == "docs"
