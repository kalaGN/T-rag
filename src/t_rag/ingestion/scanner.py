from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path

from t_rag.config import SourceConfig


SUPPORTED_EXTENSIONS = {".md", ".mdc", ".csv", ".xlsx", ".pdf", ".txt"}


@dataclass(slots=True)
class ScannedFile:
    source_name: str
    absolute_path: Path
    relative_path: Path
    file_hash: str
    size_bytes: int


def scan_source(source: SourceConfig) -> list[ScannedFile]:
    files: list[ScannedFile] = []
    root = source.root
    if not root.exists():
        return files

    for include_dir in source.include:
        include_root = root / include_dir
        if not include_root.exists():
            continue
        for path in include_root.rglob("*"):
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("数据源包含指向目录外的文件。")
            relative_path = path.relative_to(root)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if _is_excluded(relative_path, source.exclude_patterns):
                continue
            files.append(
                ScannedFile(
                    source_name=source.name,
                    absolute_path=path,
                    relative_path=relative_path,
                    file_hash=_sha1(path),
                    size_bytes=path.stat().st_size,
                )
            )
    unique = {item.relative_path.as_posix(): item for item in files}
    return [unique[key] for key in sorted(unique)]


def _is_excluded(relative_path: Path, patterns: list[str]) -> bool:
    candidate = relative_path.as_posix()
    lower_candidate = candidate.lower()
    for pattern in patterns:
        lower_pattern = pattern.lower()
        if fnmatch.fnmatch(lower_candidate, lower_pattern):
            return True
        if "**" in lower_pattern:
            if _match_double_wildcard(lower_candidate, lower_pattern):
                return True
    return False


def _match_double_wildcard(candidate: str, pattern: str) -> bool:
    prefix, _, suffix = pattern.partition("**")
    if not candidate.startswith(prefix):
        return False
    if suffix and not candidate.endswith(suffix):
        return False
    return True


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while chunk := file.read(8192):
            digest.update(chunk)
    return digest.hexdigest()
