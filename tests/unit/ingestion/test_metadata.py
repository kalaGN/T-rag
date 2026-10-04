from pathlib import Path
from t_rag.config import SourceConfig
from t_rag.ingestion.metadata import build_metadata


def test_metadata_is_generic():
    source = SourceConfig("source-1", Path("/tmp"), ["."], "")
    metadata = build_metadata(source, Path("300007-cucc.md"), "产品 300007 使用 cucc 渠道")
    assert metadata == {"source_id": "source-1", "file_path": "300007-cucc.md", "title": "300007-cucc.md", "format": "md"}
