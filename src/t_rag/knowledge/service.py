from __future__ import annotations

import hashlib
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path

from t_rag.config import Settings, SourceConfig
from t_rag.ingestion.scanner import SUPPORTED_EXTENSIONS, scan_source
from t_rag.storage.catalog import LOCK, read_manifest, save_manifest, validate_id

INDEX_FIELDS = ("embedding_base_url", "embedding_model", "embedding_dim", "chunk_sizes", "chunk_overlap")
STATUS_LABELS = {"empty": "未建索引", "dirty": "需要重建", "building": "重建中", "ready": "可用", "failed": "失败", "deleting": "删除中"}


class KnowledgeService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.data_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kb_id: str) -> Path:
        path = self.root / validate_id(kb_id)
        if path.is_symlink():
            raise ValueError("知识库目录不能是符号链接。")
        return path

    def load(self, kb_id: str) -> dict:
        with LOCK:
            item = read_manifest(self.path(kb_id) / "manifest.json")
            if item["id"] != kb_id:
                raise ValueError("知识库标识不匹配。")
            return item

    def save(self, item: dict) -> None:
        item["updated_at"] = time.time()
        save_manifest(self.path(item["id"]) / "manifest.json", item)

    def list(self) -> list[dict]:
        with LOCK:
            return sorted([self.load(p.name) for p in self.root.iterdir()
                           if p.is_dir() and len(p.name) == 32 and not p.is_symlink()],
                          key=lambda item: item["created_at"])

    def create(self, name: str) -> dict:
        with LOCK:
            name = name.strip()
            if not name or len(name) > 120:
                raise ValueError("名称需要为 1–120 个字符。")
            if any(item["name"] == name for item in self.list()):
                raise ValueError("知识库名称已存在。")
            item = {"id": uuid.uuid4().hex, "name": name, "sources": [], "status": "empty",
                    "error": "", "files": [], "index_config": {}, "node_count": 0,
                    "created_at": time.time()}
            self.save(item)
            return item

    def recover(self) -> None:
        """Called once at application startup, never during individual requests."""
        with LOCK:
            for item in self.list():
                if item["status"] == "building":
                    item.update(status="failed", error="上次重建被中断，请重新重建。")
                    self.save(item)

    def add_directory(self, kb_id: str, directory: str, include=None, exclude=None) -> dict:
        with LOCK:
            item = self.load(kb_id)
            root = Path(directory).expanduser().resolve()
            if not root.is_dir():
                raise ValueError("目录不存在。")
            include = include or ["."]
            for part in include:
                if not (root / part).resolve().is_relative_to(root):
                    raise ValueError("包含目录不能超出数据源目录。")
            exclude = exclude or [".git/**", ".DS_Store"]
            for source in item["sources"]:
                if source["kind"] == "directory" and source["root"] == str(root):
                    source.update(include=include, exclude=exclude)
                    break
            else:
                source = {"id": uuid.uuid4().hex, "kind": "directory", "root": str(root),
                          "include": include, "exclude": exclude, "name": root.name}
                item["sources"].append(source)
            item.update(status="dirty", error="")
            self.save(item)
            return source

    def upload(self, kb_id: str, files: list[str]) -> dict:
        with LOCK:
            item = self.load(kb_id)
            paths = [Path(p) for p in files]
            if not paths:
                raise ValueError("请选择文件。")
            for p in paths:
                if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    raise ValueError(f"不支持的文件：{p.name}")
            source = next((s for s in item["sources"] if s["kind"] == "upload"), None)
            if source is None:
                source = {"id": uuid.uuid4().hex, "kind": "upload", "name": "上传文件",
                          "include": ["."], "exclude": []}
                item["sources"].append(source)
            target = self.path(kb_id) / "uploads" / validate_id(source["id"])
            if not target.resolve().is_relative_to(self.path(kb_id).resolve()):
                raise ValueError("上传目录不属于知识库。")
            target.mkdir(parents=True, exist_ok=True)
            # Persist dirty state before modifying any indexed managed content.
            item.update(status="dirty", error="")
            self.save(item)
            for p in paths:
                destination = target / p.name
                if destination.resolve().parent != target.resolve():
                    raise ValueError("上传路径不属于知识库。")
                shutil.copyfile(p, destination)
            return source

    def remove_source(self, kb_id: str, source_id: str) -> None:
        with LOCK:
            validate_id(source_id)
            item = self.load(kb_id)
            source = next((s for s in item["sources"] if s["id"] == source_id), None)
            if source is None:
                raise ValueError("数据源不存在。")
            # Keep managed copies until the whole KB is deleted; no original file is removed.
            item["sources"].remove(source)
            item.update(status="dirty", error="")
            self.save(item)

    def source_root(self, kb_id: str, source: dict) -> Path:
        if source["kind"] == "upload":
            return self.path(kb_id) / "uploads" / validate_id(source["id"])
        return Path(source["root"])

    def snapshot(self, kb_id: str) -> list[dict]:
        item = self.load(kb_id)
        result = []
        for source in item["sources"]:
            root = self.source_root(kb_id, source).resolve()
            if not root.is_dir():
                raise ValueError(f"数据源目录不可访问：{source['name']}")
            config = SourceConfig(source["id"], root, source["include"], "", source["exclude"])
            for scanned in scan_source(config):
                absolute = scanned.absolute_path.resolve()
                if not absolute.is_relative_to(root):
                    raise ValueError("数据源包含指向目录外的文件。")
                relative = scanned.relative_path.as_posix()
                result.append({"id": hashlib.sha256(f"{source['id']}:{relative}".encode()).hexdigest(),
                               "source_id": source["id"], "path": relative, "hash": scanned.file_hash,
                               "absolute_path": str(absolute)})
        return sorted(result, key=lambda f: f["id"])

    def signature(self) -> dict:
        return {key: getattr(self.settings, key) for key in INDEX_FIELDS}

    def index_settings(self, kb_id: str) -> Settings:
        return replace(self.settings, qdrant_collection=f"t_rag_{validate_id(kb_id)}",
                       docstore_path=str(self.path(kb_id) / "docstore"))

    def ready(self, kb_id: str) -> dict:
        item = self.load(kb_id)
        if item["status"] != "ready":
            raise ValueError(f"知识库{STATUS_LABELS.get(item['status'], item['status'])}。请先重建。")
        try:
            current_files = self.snapshot(kb_id)
        except (ValueError, OSError) as exc:
            item.update(status="dirty", error="数据源无法访问，请检查来源后重建。")
            self.save(item)
            raise ValueError(item["error"]) from exc
        if item["index_config"] != self.signature() or item["files"] != current_files:
            item.update(status="dirty", error="文档或索引配置已变化，请重建。")
            self.save(item)
            raise ValueError(item["error"])
        return item

    def preview(self, kb_id: str, document_id: str) -> str:
        with LOCK:
            item = self.ready(kb_id)
            file = next((f for f in item["files"] if f["id"] == document_id), None)
            if file is None:
                raise ValueError("来源未登记在当前知识库。")
            from t_rag.ingestion.readers import read_sections
            return "\n\n".join(f"{section.location}\n{section.text}" for section in read_sections(Path(file["absolute_path"])))

    def delete(self, kb_id: str, confirmed: bool = False) -> None:
        if not confirmed:
            raise ValueError("删除知识库需要确认。")
        with LOCK:
            item = self.load(kb_id)
            item.update(status="deleting", error="")
            self.save(item)
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(url=self.settings.qdrant_url)
                try:
                    collection = self.index_settings(kb_id).qdrant_collection
                    if client.collection_exists(collection):
                        client.delete_collection(collection)
                finally:
                    client.close()
                shutil.rmtree(self.path(kb_id))
            except Exception as exc:
                item.update(status="failed", error="删除失败，请重试。")
                self.save(item)
                raise RuntimeError("删除失败，请检查 Qdrant 和数据目录后重试。") from exc
