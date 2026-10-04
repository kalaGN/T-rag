from t_rag.retrieval.query_rewrite import normalize_query


def test_queries_have_no_business_expansion():
    assert normalize_query("  账期  ") == "账期"
    assert normalize_query(" unknown ") == "unknown"
