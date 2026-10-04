from pathlib import Path
from t_rag.config import SourceConfig


def build_metadata(source: SourceConfig, relative_path: Path, text: str) -> dict:
    return {"source_id": source.name, "file_path": relative_path.as_posix(),
            "title": relative_path.name, "format": relative_path.suffix.lower().lstrip(".")}
