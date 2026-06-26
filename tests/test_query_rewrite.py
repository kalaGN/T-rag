from fin_rag.retrieval.query_rewrite import normalize_query


def test_normalize_query_appends_known_synonym():
    normalized = normalize_query("账期怎么切")

    assert "账期怎么切" in normalized
    assert "结算周期" in normalized


def test_normalize_query_keeps_unknown_query():
    normalized = normalize_query("LanChen 接口文档")

    assert normalized == "LanChen 接口文档"
