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


def _install_llama_index_mocks(monkeypatch) -> None:
    """安装 llama_index 模块 mock，供 workflow 测试共用。"""
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


def test_ask_question_returns_answer_when_citations_are_valid(monkeypatch):
    captured = {}
    _install_llama_index_mocks(monkeypatch)

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
    assert result.validation["refusal_reason"] is None
    assert result.raw_answer == "账期切换见 [1]"
    assert result.sources[0]["metadata"]["domain"] == "fin-online"
    assert captured["domains"] == ["fin-online"]
    assert captured["question"] == "账期切换逻辑是什么"
    assert len(captured["node_postprocessors"]) == 1
    assert captured["node_postprocessors"][0].similarity_cutoff == 0.45


def test_ask_question_refuses_when_validation_fails(monkeypatch):
    _install_llama_index_mocks(monkeypatch)

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
    assert result.validation["refusal_reason"] == "invalid_refs"
    assert result.raw_answer == "答案见 [2]，产品 300008。"


def test_ask_question_uses_lower_cutoff_for_short_query(monkeypatch):
    captured = {}
    _install_llama_index_mocks(monkeypatch)

    monkeypatch.setattr("fin_rag.qa.workflow.build_llm", lambda settings: object())

    def fake_build_query_engine(settings, domains=None, llm=None, node_postprocessors=None):
        captured["node_postprocessors"] = node_postprocessors

        class _Engine:
            def query(self, question):
                return _FakeResponse("账期见 [1]", [_source_node("账期切换见产品 300007。")])

        return _Engine()

    monkeypatch.setattr("fin_rag.qa.workflow.build_query_engine", fake_build_query_engine)

    result = ask_question(Settings(llm_base_url="http://example.test/v1"), "账期")

    assert result.refused is False
    assert captured["node_postprocessors"][0].similarity_cutoff == 0.15


def test_ask_question_retries_without_gate_when_first_query_has_no_sources(monkeypatch):
    captured = {"calls": []}
    _install_llama_index_mocks(monkeypatch)

    monkeypatch.setattr("fin_rag.qa.workflow.build_llm", lambda settings: object())

    def fake_build_query_engine(settings, domains=None, llm=None, node_postprocessors=None):
        captured["calls"].append(node_postprocessors)

        class _Engine:
            def query(self, question):
                if len(captured["calls"]) == 1:
                    return _FakeResponse("Empty Response", [])
                return _FakeResponse("opdata 对接渠道见 [1]", [_source_node("opdata 对接渠道说明。", domain="opdata")])

        return _Engine()

    monkeypatch.setattr("fin_rag.qa.workflow.build_query_engine", fake_build_query_engine)

    result = ask_question(Settings(llm_base_url="http://example.test/v1"), "opdata对接的渠道")

    assert result.refused is False
    assert result.answer == "opdata 对接渠道见 [1]"
    assert result.validation["refusal_reason"] is None
    assert len(captured["calls"]) == 2
    assert captured["calls"][0][0].similarity_cutoff == 0.45
    assert captured["calls"][1] is None
