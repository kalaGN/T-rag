from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from t_rag.config import Settings
from t_rag.knowledge.service import KnowledgeService
from t_rag.models.embedding import build_embedding
from t_rag.storage.catalog import LOCK
from t_rag.storage.runtime import build_docstore


@dataclass(slots=True)
class Hit:
    id: str
    text: str
    metadata: dict
    score: float
    scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "metadata": self.metadata,
                "score": self.score, "scores": self.scores}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    # Bigrams preserve useful Chinese word fragments alongside individual characters.
    words += re.findall(r"(?=([\u4e00-\u9fff]{2}))", text)
    return words


def bm25_search(nodes: list, question: str, top_k: int) -> list[Hit]:
    tokens = [Counter(tokenize(node.text)) for node in nodes]
    if not tokens:
        return []
    lengths = [sum(counts.values()) for counts in tokens]
    average = sum(lengths) / len(lengths) or 1
    query = set(tokenize(question))
    frequencies = {term: sum(term in counts for counts in tokens) for term in query}
    hits = []
    for node, counts, length in zip(nodes, tokens, lengths):
        score = 0.0
        for term in query:
            frequency = counts[term]
            if frequency:
                idf = math.log(1 + (len(nodes) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            hits.append(Hit(node.node_id, node.text, node.metadata, score, {"bm25": score}))
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def _retrieve(settings: Settings, knowledge_base_id: str, question: str, strategy: str = "fusion") -> list[Hit]:
    if strategy not in {"vector", "bm25", "fusion"}:
        raise ValueError("检索策略必须是 vector、bm25 或 fusion。")
    question = question.strip()
    if not question:
        raise ValueError("请输入问题。")
    with LOCK:
        service = KnowledgeService(settings)
        item = service.ready(knowledge_base_id)
        if not item["node_count"]:
            return []
        active = service.index_settings(knowledge_base_id)
        docstore = build_docstore(active)
        nodes = list(docstore.docs.values())
        valid_docs = {file["id"] for file in item["files"]}
        def belongs(metadata):
            return metadata.get("knowledge_base_id") == knowledge_base_id and metadata.get("document_id") in valid_docs
        nodes = [node for node in nodes if belongs(node.metadata)]
        if len(nodes) != item["node_count"]:
            raise RuntimeError("DocStore 与知识库不一致，请重建。")
        lists = []
        if strategy in {"vector", "fusion"}:
            from qdrant_client import QdrantClient, models
            from llama_index.core.vector_stores.utils import metadata_dict_to_node
            vector = build_embedding(active).get_query_embedding(question)
            client = QdrantClient(url=active.qdrant_url)
            try:
                points = client.query_points(collection_name=active.qdrant_collection, query=vector,
                    query_filter=models.Filter(must=[models.FieldCondition(key="knowledge_base_id", match=models.MatchValue(value=knowledge_base_id))]),
                    limit=active.fusion_top_k, score_threshold=active.similarity_cutoff, with_payload=True).points
                vector_hits = []
                for point in points:
                    node = metadata_dict_to_node(point.payload)
                    if belongs(node.metadata):
                        vector_hits.append(Hit(node.node_id, node.text, node.metadata, point.score, {"vector": point.score}))
                lists.append(vector_hits)
            finally:
                client.close()
        if strategy in {"bm25", "fusion"}:
            lists.append(bm25_search(nodes, question, active.fusion_top_k))
        if strategy != "fusion":
            return lists[0]
        combined = {}
        for hits in lists:
            for rank, hit in enumerate(hits, 1):
                if hit.id not in combined:
                    combined[hit.id] = Hit(hit.id, hit.text, hit.metadata, 0.0)
                combined[hit.id].score += 1 / (60 + rank)
                combined[hit.id].scores.update(hit.scores)
        return sorted(combined.values(), key=lambda hit: hit.score, reverse=True)[:active.fusion_top_k]


def retrieve(settings: Settings, knowledge_base_id: str, question: str, strategy: str = "fusion") -> list[Hit]:
    try:
        return _retrieve(settings, knowledge_base_id, question, strategy)
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError("检索失败，请检查 Qdrant、Embedding 服务和索引是否完整。") from exc
