from types import SimpleNamespace
from t_rag.retrieval.fusion import bm25_search


def test_bm25_chinese_and_unknown_question():
    nodes = [SimpleNamespace(node_id="1", text="抹茶蛋糕需要冷藏", metadata={}),
             SimpleNamespace(node_id="2", text="网络请求超时", metadata={})]
    hits = bm25_search(nodes, "蛋糕", 8)
    assert [hit.id for hit in hits] == ["1"]
    assert hits[0].scores["bm25"] > 0
    assert bm25_search(nodes, "火星殖民", 8) == []
