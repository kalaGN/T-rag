from t_rag.qa.citation import CitationContext, validate_citations
from t_rag.qa.prompts import build_system_prompt, REFUSAL_TEMPLATE


def test_generic_prompt_and_reference_rules():
    contexts = [CitationContext(1, "kb", "a.md", "章节", "原文")]
    assert REFUSAL_TEMPLATE in build_system_prompt("kb", contexts)
    assert validate_citations("数值 300007 [1]", contexts)["valid"]
    assert validate_citations("没有引用", contexts)["refusal_reason"] == "missing_refs"
    assert validate_citations("错误引用 [2]", contexts)["refusal_reason"] == "invalid_refs"
    assert validate_citations("", contexts)["refusal_reason"] == "empty_answer"
    assert validate_citations("[1]", [])["refusal_reason"] == "no_sources"
