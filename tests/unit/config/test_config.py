import sys
from types import ModuleType
from pathlib import Path

import pytest

from t_rag.config import Settings
from t_rag.storage.runtime import build_docstore


def test_build_docstore_loads_existing_persisted_docstore(tmp_path: Path):
    settings = Settings(docstore_path=str(tmp_path))
    fake_module = ModuleType("llama_index")
    fake_core = ModuleType("llama_index.core")
    fake_storage = ModuleType("llama_index.core.storage")
    fake_docstore = ModuleType("llama_index.core.storage.docstore")

    class SimpleDocumentStore:
        def __init__(self):
            self.persisted_path = None

        @classmethod
        def from_persist_path(cls, path: str):
            instance = cls()
            instance.persisted_path = path
            return instance

        def persist(self, path: str):
            Path(path).write_text("{}", encoding="utf-8")

    fake_docstore.SimpleDocumentStore = SimpleDocumentStore
    fake_storage.docstore = fake_docstore
    fake_core.storage = fake_storage
    fake_module.core = fake_core

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "llama_index", fake_module)
    monkeypatch.setitem(sys.modules, "llama_index.core", fake_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.storage", fake_storage)
    monkeypatch.setitem(sys.modules, "llama_index.core.storage.docstore", fake_docstore)
    try:
        build_docstore().persist(str(tmp_path / "docstore.json"))

        loaded = build_docstore(settings)
    finally:
        monkeypatch.undo()

    assert loaded is not None
    assert loaded.persisted_path == str(tmp_path / "docstore.json")


def test_new_env_prefix_and_validation(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("embedding_dim: 768\n")
    monkeypatch.setenv("FIN_RAG_EMBEDDING_DIM", "1")
    assert Settings.from_yaml(settings_file).embedding_dim == 768
    monkeypatch.setenv("T_RAG_EMBEDDING_DIM", "128")
    assert Settings.from_yaml(settings_file).embedding_dim == 128
    monkeypatch.setenv("T_RAG_EMBEDDING_DIM", "0")
    with pytest.raises(ValueError): Settings.from_yaml(settings_file)
