from t_rag.knowledge.service import KnowledgeService, STATUS_LABELS
from t_rag.knowledge.indexing import rebuild
from t_rag.storage.catalog import LOCK


class Management:
    def __init__(self, settings):
        self.service = KnowledgeService(settings)
        self.service.recover()

    def choices(self):
        return [(item["name"], item["id"]) for item in self.service.list()]

    def details(self, kb_id):
        if not kb_id:
            return "请创建或选择知识库。", []
        with LOCK:
            item = self.service.load(kb_id)
            if item["status"] == "ready":
                try:
                    item = self.service.ready(kb_id)
                except ValueError:
                    item = self.service.load(kb_id)
            sources = [(s["name"], s["id"]) for s in item["sources"]]
            status = STATUS_LABELS.get(item["status"], item["status"])
            summary = f"状态：{status} · {len(item['sources'])} 个来源 · {item['node_count']} 个已索引片段"
            if item["error"]:
                summary += f"\n\n{item['error']}"
            return summary, sources

    def create(self, name):
        return self.service.create(name)["id"]

    def upload(self, kb_id, files):
        self.service.upload(kb_id, files or [])
        return "文件已导入，请重建索引。"

    def directory(self, kb_id, directory, includes, excludes):
        include = [v.strip() for v in includes.splitlines() if v.strip()] or ["."]
        exclude = [v.strip() for v in excludes.splitlines() if v.strip()]
        self.service.add_directory(kb_id, directory, include, exclude)
        return "目录已登记，请重建索引。"

    def remove(self, kb_id, source_id):
        self.service.remove_source(kb_id, source_id)
        return "来源已移除，请重建索引。原始文件保持不变。"

    def rebuild(self, kb_id, progress=None):
        item = rebuild(self.service, kb_id, progress)
        return f"重建完成：{len(item['files'])} 个文档，{item['node_count']} 个片段。"

    def delete(self, kb_id, confirmed):
        self.service.delete(kb_id, confirmed)
