from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from t_rag.config import Settings, SourceConfig
from t_rag.ingestion.readers import LoadedDocument, load_document
from t_rag.ingestion.scanner import ScannedFile, scan_source


@dataclass(slots=True)
class CollectedDocuments:
    documents: list[LoadedDocument]
    scanned_files: list[ScannedFile]


def collect_loaded_documents(settings: Settings, sources: list[SourceConfig]) -> CollectedDocuments:
    documents: list[LoadedDocument] = []
    scanned_files: list[ScannedFile] = []

    for source in sources:
        source_files = scan_source(source)
        scanned_files.extend(source_files)
        for scanned_file in source_files:
            loaded = load_document(
                path=scanned_file.absolute_path,
                relative_path=scanned_file.relative_path,
                source=source,
                settings=settings,
                file_hash=scanned_file.file_hash,
                size_bytes=scanned_file.size_bytes,
            )
            loaded.metadata.update(
                {
                    "doc_id": f"{source.name}:{loaded.relative_path}",
                    "file_hash": scanned_file.file_hash,
                    "size_bytes": str(scanned_file.size_bytes),
                }
            )
            documents.append(loaded)
    return CollectedDocuments(documents=documents, scanned_files=scanned_files)


def to_llama_documents(documents: list[LoadedDocument]) -> list[Any]:
    try:
        from llama_index.core import Document
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index is not installed. Install project dependencies before running real indexing."
        ) from exc

    return [
        Document(
            text=document.text,
            metadata=document.metadata,
            doc_id=str(document.metadata["doc_id"]),
        )
        for document in documents
    ]
