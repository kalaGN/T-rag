import sys
from types import ModuleType

from fin_rag.config import Settings
from fin_rag.retrieval.query_engine import build_query_engine


def test_build_query_engine_passes_llm_and_retriever(monkeypatch):
    retriever = object()
    llm = object()

    fake_module = ModuleType("llama_index.core.query_engine")
    root_module = ModuleType("llama_index")
    root_module.__path__ = []
    core_module = ModuleType("llama_index.core")
    core_module.__path__ = []
    monkeypatch.setitem(sys.modules, "llama_index", root_module)
    monkeypatch.setitem(sys.modules, "llama_index.core", core_module)

    class CitationQueryEngine:
        last_kwargs = None

        @classmethod
        def from_args(cls, **kwargs):
            cls.last_kwargs = kwargs
            return kwargs

    fake_module.CitationQueryEngine = CitationQueryEngine
    monkeypatch.setitem(sys.modules, "llama_index.core.query_engine", fake_module)
    def fake_build_retriever(settings, domains=None, llm=None, index=None):
        assert llm is llm_value
        assert index is index_value
        return retriever

    llm_value = llm
    index_value = object()
    monkeypatch.setattr("fin_rag.retrieval.query_engine.build_retriever", fake_build_retriever)
    monkeypatch.setattr("fin_rag.retrieval.query_engine.build_vector_index", lambda settings: index_value)

    settings = Settings(fusion_top_k=5, llm_base_url="http://example.test/v1")
    result = build_query_engine(settings, domains=["fin-online"], llm=llm)

    assert result["llm"] is llm
    assert result["retriever"] is retriever
    assert result["index"] is index_value
    assert result["similarity_top_k"] == 5
    assert CitationQueryEngine.last_kwargs["llm"] is llm
