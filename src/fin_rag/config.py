from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"


@dataclass(slots=True)
class SourceConfig:
    name: str
    root: Path
    include: list[str]
    default_domain: str
    exclude_patterns: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceConfigFile:
    sources: list[SourceConfig]


@dataclass(slots=True)
class Settings:
    app_name: str = "fin-rag"
    qdrant_collection: str = "fin_rag_docs"
    qdrant_url: str = "http://localhost:6333"
    llm_base_url: str = ""
    llm_model: str = "deepseek-chat"
    llm_api_key: str | None = None
    llm_timeout: float = 30.0
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    embedding_base_url: str = "http://localhost:8001/v1"
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_api_key: str | None = None
    embedding_dim: int = 768
    embedding_batch_size: int = 8
    embedding_host: str = "127.0.0.1"
    embedding_port: int = 8001
    chunk_sizes: list[int] = field(default_factory=lambda: [1024, 512, 256])
    chunk_overlap: int = 80
    similarity_cutoff: float = 0.45
    fusion_top_k: int = 8
    rerank_top_n: int = 6
    large_file_threshold_bytes: int = 1_048_576
    docstore_path: str = "data/docstore"

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "Settings":
        data = _load_yaml(path or CONFIG_DIR / "settings.yaml")
        merged = _apply_env_overrides(data)
        return cls(**merged)


def load_sources(path: Path | None = None) -> list[SourceConfig]:
    data = _load_yaml(path or CONFIG_DIR / "sources.yaml")
    sources = []
    for item in data.get("sources", []):
        source_data = dict(item)
        source_data["root"] = Path(source_data["root"])
        sources.append(SourceConfig(**source_data))
    return SourceConfigFile(sources=sources).sources


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    for key, caster in _SETTING_CASTERS.items():
        env_key = f"FIN_RAG_{key.upper()}"
        if env_key not in os.environ:
            continue
        raw_value = os.environ[env_key]
        try:
            merged[key] = caster(raw_value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid value for environment variable {env_key}: {raw_value!r}"
            ) from exc
    return merged


_SETTING_CASTERS = {
    "app_name": str,
    "qdrant_collection": str,
    "qdrant_url": str,
    "llm_base_url": str,
    "llm_model": str,
    "llm_api_key": str,
    "llm_timeout": float,
    "llm_max_tokens": int,
    "llm_temperature": float,
    "embedding_base_url": str,
    "embedding_model": str,
    "embedding_api_key": str,
    "embedding_dim": int,
    "embedding_batch_size": int,
    "embedding_host": str,
    "embedding_port": int,
    "chunk_overlap": int,
    "similarity_cutoff": float,
    "fusion_top_k": int,
    "rerank_top_n": int,
    "large_file_threshold_bytes": int,
    "docstore_path": str,
    "chunk_sizes": lambda value: [int(part.strip()) for part in value.split(",") if part.strip()],
}
