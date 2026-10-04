from __future__ import annotations

from t_rag.knowledge.service import KnowledgeService
from t_rag.storage.catalog import LOCK
from t_rag.ingestion.llama_pipeline import build_nodes, write_index


def rebuild(service: KnowledgeService, kb_id: str, progress=None) -> dict:
    with LOCK:
        item = service.load(kb_id)
        item.update(status="building", error="")
        service.save(item)
        try:
            if progress:
                progress(0.1, desc="扫描与解析文档")
            snapshot = service.snapshot(kb_id)
            signature = service.signature()
            settings = service.index_settings(kb_id)
            nodes = build_nodes(settings, kb_id, snapshot)
            if progress:
                progress(0.4, desc=f"正在索引 {len(nodes)} 个片段")
            write_index(settings, nodes)
            if service.snapshot(kb_id) != snapshot or service.signature() != signature:
                raise RuntimeError("文档在索引期间发生变化，请重新重建。")
            item.update(status="ready", files=snapshot, index_config=signature, node_count=len(nodes), error="")
            service.save(item)
            if progress:
                progress(1, desc="索引完成")
            return item
        except Exception as exc:
            # Detailed input filenames are safe; never include credentials or HTTP response bodies.
            detail = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}，请检查本地 Qdrant / Embedding 服务"
            item.update(status="failed", error=f"重建失败：{detail}。可再次重建。")
            service.save(item)
            raise RuntimeError(item["error"]) from exc
