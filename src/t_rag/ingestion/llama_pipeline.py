from __future__ import annotations

from pathlib import Path
from t_rag.config import Settings
from t_rag.ingestion.readers import read_sections
from t_rag.models.embedding import build_embedding


def build_nodes(settings: Settings, kb_id: str, files: list[dict]):
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=min(settings.chunk_sizes), chunk_overlap=settings.chunk_overlap)
    nodes = []
    for file in files:
        try:
            sections = read_sections(Path(file["absolute_path"]))
        except Exception as exc:
            raise ValueError(f"文档解析失败：{file['path']}（{type(exc).__name__}）；检查格式、编码或是否需要 OCR。") from exc
        for section_index, section in enumerate(sections):
            document = Document(text=section.text, id_=f"{file['id']}:{section_index}", metadata={
                "knowledge_base_id": kb_id, "document_id": file["id"], "source_id": file["source_id"],
                "file_path": file["path"], "location": section.location, "file_hash": file["hash"]},
                excluded_embed_metadata_keys=["knowledge_base_id", "document_id", "source_id", "file_hash"],
                excluded_llm_metadata_keys=["knowledge_base_id", "document_id", "source_id", "file_hash"])
            nodes.extend(splitter.get_nodes_from_documents([document]))
    return nodes


def write_index(settings: Settings, nodes: list) -> None:
    from qdrant_client import QdrantClient, models
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from t_rag.storage.runtime import build_qdrant_vector_store
    client = QdrantClient(url=settings.qdrant_url)
    try:
        if client.collection_exists(settings.qdrant_collection):
            client.delete_collection(settings.qdrant_collection)
        client.create_collection(settings.qdrant_collection,
                                 vectors_config=models.VectorParams(size=settings.embedding_dim, distance=models.Distance.COSINE))
    finally:
        client.close()
    if nodes:
        vectors = build_embedding(settings).get_text_embedding_batch([n.get_content(metadata_mode="embed") for n in nodes])
        if len(vectors) != len(nodes) or any(len(vector) != settings.embedding_dim for vector in vectors):
            raise ValueError("Embedding 返回数量或维度不匹配。")
        for node, vector in zip(nodes, vectors):
            node.embedding = vector
        store = build_qdrant_vector_store(settings)
        try:
            store.add(nodes)
            if store.client.count(collection_name=settings.qdrant_collection, exact=True).count != len(nodes):
                raise RuntimeError("向量数量校验失败。")
        finally:
            store.client.close()
    docstore = SimpleDocumentStore()
    # The DocStore contains the exact same leaf chunks as Qdrant for BM25.
    docstore.add_documents(nodes)
    target = Path(settings.docstore_path)
    target.mkdir(parents=True, exist_ok=True)
    docstore.persist(str(target / "docstore.json"))
    if len(SimpleDocumentStore.from_persist_path(str(target / "docstore.json")).docs) != len(nodes):
        raise RuntimeError("DocStore 数量校验失败。")
