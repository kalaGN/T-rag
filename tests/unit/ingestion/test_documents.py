from pathlib import Path

from t_rag.config import Settings, SourceConfig
from t_rag.ingestion.documents import collect_loaded_documents


def test_collect_loaded_documents_adds_doc_id_and_hash(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    file_path = root / "docs" / "a.md"
    file_path.write_text("hello 300007", encoding="utf-8")

    source = SourceConfig(
        name="fin-online",
        root=root,
        include=["docs"],
        default_domain="fin-online",
        exclude_patterns=[],
    )
    settings = Settings()

    collected = collect_loaded_documents(settings=settings, sources=[source])

    assert len(collected.documents) == 1
    document = collected.documents[0]
    assert document.metadata["doc_id"] == "fin-online:docs/a.md"
    assert document.metadata["file_hash"] == document.file_hash
    assert document.metadata["size_bytes"] == str(document.size_bytes)
