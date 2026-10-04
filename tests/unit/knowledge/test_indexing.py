import pytest
from t_rag.config import Settings
from t_rag.knowledge.service import KnowledgeService
from t_rag.knowledge.indexing import rebuild


def test_failure_blocks_query_and_retry_recovers(tmp_path, monkeypatch):
    service = KnowledgeService(Settings(data_dir=str(tmp_path)))
    kb = service.create("A")["id"]
    monkeypatch.setattr("t_rag.knowledge.indexing.build_nodes", lambda *args: [])
    def fail(*args): raise OSError("write failed")
    monkeypatch.setattr("t_rag.knowledge.indexing.write_index", fail)
    with pytest.raises(RuntimeError): rebuild(service, kb)
    assert service.load(kb)["status"] == "failed"
    with pytest.raises(ValueError): service.ready(kb)
    monkeypatch.setattr("t_rag.knowledge.indexing.write_index", lambda *args: None)
    assert rebuild(service, kb)["status"] == "ready"


def test_file_change_during_build_is_not_published(tmp_path, monkeypatch):
    service = KnowledgeService(Settings(data_dir=str(tmp_path / "data")))
    kb = service.create("A")["id"]
    source = tmp_path / "source"; source.mkdir()
    file = source / "a.txt"; file.write_text("first")
    service.add_directory(kb, str(source))
    monkeypatch.setattr("t_rag.knowledge.indexing.build_nodes", lambda *args: [])
    monkeypatch.setattr("t_rag.knowledge.indexing.write_index", lambda *args: file.write_text("second"))
    with pytest.raises(RuntimeError): rebuild(service, kb)
    assert service.load(kb)["status"] == "failed"
