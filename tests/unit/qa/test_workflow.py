from types import SimpleNamespace
import pytest
from t_rag.config import Settings
from t_rag.retrieval.fusion import Hit
from t_rag.qa.workflow import ask_question


@pytest.mark.parametrize("text,refused", [("结论 [1]", False), ("结论", True), ("结论 [9]", True), ("", True)])
def test_answer_citations(monkeypatch, text, refused):
    hit = Hit("a", "文档", {"file_path": "a.md", "location": "标题", "document_id": "a"}, 0.9)
    monkeypatch.setattr("t_rag.qa.workflow.retrieve", lambda *args: [hit])
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=text))
    result = ask_question(Settings(), "问题", "a" * 32, llm)
    assert result.refused is refused
    assert result.sources[0]["index"] == 1


def test_no_candidates_never_calls_llm(monkeypatch):
    monkeypatch.setattr("t_rag.qa.workflow.retrieve", lambda *args: [])
    def fail(*args):
        raise AssertionError("must not call model")
    result = ask_question(Settings(), "问题", "a" * 32, SimpleNamespace(complete=fail))
    assert result.refused
    assert result.validation["refusal_reason"] == "no_sources"


def test_model_failure_has_actionable_redacted_message(monkeypatch):
    hit = Hit("a", "文档", {"file_path": "a.md", "location": "标题", "document_id": "a"}, 0.9)
    monkeypatch.setattr("t_rag.qa.workflow.retrieve", lambda *args: [hit])
    def fail(*args): raise Exception("secret-provider-response")
    with pytest.raises(RuntimeError, match="回答生成失败") as exc:
        ask_question(Settings(), "问题", "a" * 32, SimpleNamespace(complete=fail))
    assert "secret-provider-response" not in str(exc.value)
