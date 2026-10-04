from __future__ import annotations

from collections import Counter

from t_rag.config import Settings, SourceConfig
from t_rag.ingestion.documents import collect_loaded_documents


class IndexPlan:
    def __init__(self, settings: Settings, sources: list[SourceConfig]) -> None:
        self.settings = settings
        self.sources = sources

    def scan(self) -> dict[str, object]:
        source_summaries: list[dict[str, object]] = []
        doc_type_counter: Counter[str] = Counter()
        source_type_counter: Counter[str] = Counter()
        collected = collect_loaded_documents(settings=self.settings, sources=self.sources)
        by_source: dict[str, list] = {}
        for document in collected.documents:
            by_source.setdefault(document.source, []).append(document)

        for source in self.sources:
            source_documents = by_source.get(source.name, [])
            sample_documents = []
            for index, loaded in enumerate(source_documents):
                doc_type = str(loaded.metadata.get("doc_type"))
                source_type = str(loaded.metadata.get("source_type"))
                doc_type_counter[doc_type] += 1
                source_type_counter[source_type] += 1
                if index < 5:
                    sample_documents.append(
                        {
                            "path": loaded.relative_path,
                            "doc_type": doc_type,
                            "source_type": source_type,
                            "domain": loaded.metadata.get("domain"),
                            "summary_only": loaded.summary_only,
                        }
                    )
            source_summaries.append(
                {
                    "source": source.name,
                    "root": str(source.root),
                    "scanned_files": len(source_documents),
                    "sample_documents": sample_documents,
                }
            )

        return {
            "app": self.settings.app_name,
            "qdrant_collection": self.settings.qdrant_collection,
            "embedding_model": self.settings.embedding_model,
            "total_files": len(collected.documents),
            "doc_types": dict(doc_type_counter),
            "source_types": dict(source_type_counter),
            "sources": source_summaries,
        }


def build_index_plan(settings: Settings, sources: list[SourceConfig]) -> IndexPlan:
    return IndexPlan(settings=settings, sources=sources)
