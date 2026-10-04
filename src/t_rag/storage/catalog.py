"""Atomic JSON manifests for local knowledge bases."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

LOCK = threading.RLock()
ID_PATTERN = re.compile(r"[a-f0-9]{32}")


def validate_id(value: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError("无效的知识库或数据源 ID。")
    return value


def save_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise ValueError("知识库不存在。")
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def application_lock(root: Path):
    """Reject a second writer process for the same data directory (macOS/Linux)."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".application.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("该数据目录已有应用运行，请先停止页面或另一个 CLI 进程。") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
