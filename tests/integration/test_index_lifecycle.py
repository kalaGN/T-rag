"""Opt-in tests against actual local Qdrant and Embedding services."""
import os
from dataclasses import replace
from pathlib import Path

import pytest

from t_rag.config import Settings
from t_rag.knowledge.service import KnowledgeService
from t_rag.knowledge.indexing import rebuild
from t_rag.retrieval.fusion import retrieve

pytestmark = pytest.mark.skipif(os.getenv("T_RAG_INTEGRATION") != "1", reason="requires actual Qdrant and Embedding services")


def test_actual_services_isolation_update_remove_and_delete(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), similarity_cutoff=0.0, chunk_sizes=[256])
    service = KnowledgeService(settings)
    a = service.create("A")["id"]
    b = service.create("B")["id"]
    first = tmp_path / "first"; first.mkdir()
    second = tmp_path / "second"; second.mkdir()
    file = first / "same.md"
    file.write_text("# 蛋糕储存\n抹茶蛋糕需要冷藏，保质期是三天。独有标记 ALPHA。")
    (second / "same.md").write_text("# 蛋糕储存\n草莓蛋糕需要冷藏，保质期是五天。独有标记 BETA。")
    source = service.add_directory(a, str(first))
    service.add_directory(b, str(second))
    try:
        rebuilt = rebuild(service, a)
        assert rebuilt["node_count"] > 0
        assert rebuild(service, a)["node_count"] == rebuilt["node_count"]
        rebuild(service, b)
        for strategy in ("vector", "bm25", "fusion"):
            hits = retrieve(settings, a, "蛋糕冷藏", strategy)
            assert hits
            assert all(hit.metadata["knowledge_base_id"] == a for hit in hits)
            assert all("BETA" not in hit.text for hit in hits)
        hits = retrieve(settings, b, "蛋糕冷藏", "fusion")
        assert hits and all("ALPHA" not in hit.text for hit in hits)
        assert service.preview(a, service.load(a)["files"][0]["id"]).startswith("蛋糕储存")
        with pytest.raises(ValueError): service.preview(a, service.load(b)["files"][0]["id"])
        with pytest.raises(ValueError, match="变化"):
            retrieve(replace(settings, embedding_model="different"), a, "蛋糕")
        rebuild(service, a)
        file.write_text("# 蛋糕储存\n抹茶蛋糕新版本，冷藏七天。独有标记 GAMMA。")
        with pytest.raises(ValueError, match="变化"): retrieve(settings, a, "蛋糕")
        rebuild(service, a)
        for strategy in ("vector", "bm25", "fusion"):
            hits = retrieve(settings, a, "蛋糕", strategy)
            assert hits and all("ALPHA" not in hit.text for hit in hits)
            assert any("GAMMA" in hit.text for hit in hits)
        service.remove_source(a, source["id"])
        with pytest.raises(ValueError): retrieve(settings, a, "蛋糕")
        rebuild(service, a)
        for strategy in ("vector", "bm25", "fusion"):
            assert retrieve(settings, a, "蛋糕", strategy) == []
        assert file.exists()
    finally:
        for kb in (a, b):
            service.delete(kb, True)
    assert file.exists()
    assert not service.path(a).exists()
