from pathlib import Path

from fin_rag.config import SourceConfig
from fin_rag.ingestion.metadata import build_metadata


def _source() -> SourceConfig:
    return SourceConfig(
        name="fin-online",
        root=Path("/tmp/fin-online"),
        include=["docs", ".cursor"],
        exclude_patterns=[],
        default_domain="fin-online",
    )


def test_build_metadata_for_cursor_rule():
    metadata = build_metadata(
        source=_source(),
        relative_path=Path(".cursor/rules/anti-hallucination.mdc"),
        text="只基于引用作答",
    )

    assert metadata["source_type"] == "cursor_rule"
    assert metadata["doc_type"] == "engineering_rule"
    assert metadata["domain"] == "fin-online"


def test_build_metadata_extracts_product_code_and_channel():
    metadata = build_metadata(
        source=_source(),
        relative_path=Path("docs/plan/300007-cucc-账期切换.md"),
        text="产品 300007 使用 cucc 渠道",
    )

    assert metadata["product_code"] == "300007"
    assert metadata["channel"] == "cucc"
