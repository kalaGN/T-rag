from pathlib import Path

from t_rag.config import SourceConfig
from t_rag.ingestion.scanner import scan_source


def test_scan_source_applies_excludes(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "keep.md").write_text("ok", encoding="utf-8")
    (root / "docs" / "testcase").mkdir()
    (root / "docs" / "testcase" / "skip.md").write_text("skip", encoding="utf-8")
    (root / "docs" / "bxf-note.md").write_text("skip", encoding="utf-8")

    source = SourceConfig(
        name="fin-online",
        root=root,
        include=["docs"],
        exclude_patterns=["docs/testcase/**", "*bxf*"],
        default_domain="fin-online",
    )

    files = scan_source(source)

    assert [item.relative_path.as_posix() for item in files] == ["docs/keep.md"]
