from pathlib import Path
import pytest
from dataclasses import replace
from t_rag.config import Settings
from t_rag.knowledge.service import KnowledgeService
from t_rag.storage.catalog import save_manifest


@pytest.fixture
def service(tmp_path):
    return KnowledgeService(Settings(data_dir=str(tmp_path / "data")))


def test_create_persist_duplicate_and_path_guard(service):
    item = service.create("通用文档")
    assert KnowledgeService(service.settings).list()[0]["id"] == item["id"]
    with pytest.raises(ValueError): service.create("通用文档")
    with pytest.raises(ValueError): service.load("../data")


def test_sources_snapshot_and_model_changes(service, tmp_path):
    kb = service.create("A")["id"]
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "bxf.md").write_text("普通文档")
    source = service.add_directory(kb, str(directory))
    first = service.snapshot(kb)
    assert len(first) == 1
    assert first[0]["path"] == "bxf.md"
    item = service.load(kb)
    item.update(status="ready", files=first, index_config=service.signature(), node_count=1)
    service.save(item)
    assert service.ready(kb)
    changed = KnowledgeService(replace(service.settings, embedding_dim=3))
    with pytest.raises(ValueError, match="变化"): changed.ready(kb)
    assert service.load(kb)["status"] == "dirty"
    service.remove_source(kb, source["id"])
    assert (directory / "bxf.md").exists()
    assert service.snapshot(kb) == []


def test_upload_replacement_and_external_file_change(service, tmp_path):
    kb = service.create("A")["id"]
    file = tmp_path / "file.txt"
    file.write_text("版本1")
    source = service.upload(kb, [str(file)])
    first = service.snapshot(kb)
    file.write_text("版本2")
    assert service.upload(kb, [str(file)])["id"] == source["id"]
    second = service.snapshot(kb)
    assert first[0]["id"] == second[0]["id"]
    assert first[0]["hash"] != second[0]["hash"]
    assert file.read_text() == "版本2"


def test_symlink_outside_root_is_rejected(service, tmp_path):
    kb = service.create("A")["id"]
    directory = tmp_path / "source"; directory.mkdir()
    external = tmp_path / "secret.txt"; external.write_text("secret")
    (directory / "escape.txt").symlink_to(external)
    service.add_directory(kb, str(directory))
    with pytest.raises(ValueError, match="目录外"): service.snapshot(kb)


def test_atomic_save_failure_preserves_previous_manifest(service, monkeypatch):
    kb = service.create("A")["id"]
    item = service.load(kb)
    item["name"] = "B"
    def fail(*args): raise OSError("disk full")
    monkeypatch.setattr("t_rag.storage.catalog.os.replace", fail)
    with pytest.raises(OSError): service.save(item)
    assert service.load(kb)["name"] == "A"


def test_recover_building_and_unregistered_preview(service):
    kb = service.create("A")["id"]
    item = service.load(kb); item["status"] = "building"; service.save(item)
    service.recover()
    assert service.load(kb)["status"] == "failed"
    with pytest.raises(ValueError): service.preview(kb, "/etc/passwd")


def test_delete_failure_retry_preserves_original(service, tmp_path, monkeypatch):
    from types import SimpleNamespace
    kb = service.create("A")["id"]
    folder = tmp_path / "docs"; folder.mkdir()
    original = folder / "manual.txt"; original.write_text("original")
    service.add_directory(kb, str(folder))
    with pytest.raises(ValueError, match="确认"): service.delete(kb)
    def fail(*args, **kwargs): raise OSError("unavailable")
    monkeypatch.setattr("qdrant_client.QdrantClient", fail)
    with pytest.raises(RuntimeError): service.delete(kb, True)
    assert service.load(kb)["status"] == "failed"
    assert original.exists()
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kwargs: SimpleNamespace(collection_exists=lambda *args: False, close=lambda: None))
    service.delete(kb, True)
    assert not service.path(kb).exists()
    assert original.read_text() == "original"


def test_manifest_empty_library_and_file_changes(service, tmp_path):
    kb = service.create("A")["id"]
    folder = tmp_path / "docs"; folder.mkdir()
    file = folder / "a.txt"; file.write_text("old")
    service.add_directory(kb, str(folder))
    item = service.load(kb)
    item.update(status="ready", files=service.snapshot(kb), index_config=service.signature(), node_count=1)
    service.save(item)
    file.write_text("new")
    with pytest.raises(ValueError, match="变化"): service.ready(kb)
    assert service.load(kb)["status"] == "dirty"


def test_application_lock_prevents_second_process(tmp_path):
    import subprocess
    import sys
    from t_rag.storage.catalog import application_lock
    root = tmp_path / "data"
    script = "from pathlib import Path; from t_rag.storage.catalog import application_lock; application_lock(Path(__import__('sys').argv[1])).__enter__()"
    with application_lock(root):
        result = subprocess.run([sys.executable, "-c", script, str(root)], env={**__import__('os').environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
        assert result.returncode != 0
        assert "已有应用运行" in result.stderr
