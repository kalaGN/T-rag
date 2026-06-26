import sys
from types import ModuleType, SimpleNamespace

from fin_rag.config import Settings
from fin_rag.qa.workflow import ask_question


class _FakeResponse:
    def __init__(self, response: str, source_nodes):
        self.response = response
        self.source_nodes = source_nodes


def _source_node(text: str, score: float = 0.9, domain: str = "fin-online", file_path: str = "docs/a.md", heading_path: str = "1"):
    return SimpleNamespace(
        score=score,
        node=SimpleNamespace(
            text=text,
            metadata={
                "domain": domain,
                "file_path": file_path,
                "heading_path": heading_path,
            },
        ),
    )


def test_ask_question_returns_answer_when_citations_are_valid(monkeypatch):
    captured = {}

    root_module = ModuleType("llama_index")
    root_module.__path__ = []
    core_module = ModuleType("llama_index.core")
    core_module.__path__ = []
    indices_module = ModuleType("llama_index.core.indices")
    indices_module.__path__ = []
    postprocessor_module = ModuleType("llama_index.core.indices.postprocessor")

    class SimilarityPostprocessor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    postprocessor_module.SimilarityPostprocessor = SimilarityPostprocessor
    monkeypatch.setitem(sys.modules, "llama_index", root_module)
    monkeypatch.setitem(sys.modules, "llama_index.core", core_module)
    monkeypatch.setitem(sys.modules, "llama_index.core.indices", indices_module)
    monkeypatch.setitem(sys.modules, "llama_index.core.indices.postprocessor", postprocessor_module)

    def fake_build_llm(settings):
        return object()

    def fake_build_query_engine(settings, domains=None, llm=None, node_postprocessors=None):
        captured["domains"] = domains
        captured["llm"] = llm
        captured["node_postprocessors"] = node_postprocessors

        class _Engine:
            def query(self, question):
                captured["question"] = question
                return _FakeResponse("账期切换见 [1]", [_source_node("账期切换见产品 300007。")])

        return _Engine()

    monkeypatch.setattr("fin_rag.qa.workflow.build_llm", fake_build_llm)
    monkeypatch.setattr("fin_rag.qa.workflow.build_query_engine", fake_build_query_engine)

    result = ask_question(Settings(llm_base_url="http://example.test/v1"), "账期切换逻辑是什么", domains=["fin-online"])

    assert result.answer == "账期切换见 [1]"
    assert result.refused is False
    assert result.validation["valid"] is True
    assert result.sources[0]["metadata"]["domain"] == "fin-online"
    assert captured["domains"] == ["fin-online"]
    assert captured["question"] == "账期切换逻辑是什么"
    assert len(captured["node_postprocessors"]) == 1


def test_ask_question_refuses_when_validation_fails(monkeypatch):
    root_module = ModuleType("llama_index")
    root_module.__path__ = []
    core_module = ModuleType("llama_index.core")
    core_module.__path__ = []
    indices_module = ModuleType("llama_index.core.indices")
    indices_module.__path__ = []
    postprocessor_module = ModuleType("llama_index.core.indices.postprocessor")

    class SimilarityPostprocessor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    postprocessor_module.SimilarityPostprocessor = SimilarityPostprocessor
    monkeypatch.setitem(sys.modules, "llama_index", root_module)
    monkeypatch.setitem(sys.modules, "llama_index.core", core_module)
    monkeypatch.setitem(sys.modules, "llama_index.core.indices", indices_module)
    monkeypatch.setitem(sys.modules, "llama_index.core.indices.postprocessor", postprocessor_module)

    def fake_build_llm(settings):
        return object()

    def fake_build_query_engine(settings, domains=None, llm=None, node_postprocessors=None):
        class _Engine:
            def query(self, question):
                return _FakeResponse("答案见 [2]，产品 300008。", [_source_node("账期切换见产品 300007。")])

        return _Engine()

    monkeypatch.setattr("fin_rag.qa.workflow.build_llm", fake_build_llm)
    monkeypatch.setattr("fin_rag.qa.workflow.build_query_engine", fake_build_query_engine)

    result = ask_question(Settings(llm_base_url="http://example.test/v1"), "账期切换逻辑是什么")

    assert result.answer == "知识库未覆盖该问题，请补充更具体的产品、渠道、文件或章节线索。"
    assert result.refused is True
    assert result.validation["valid"] is False
