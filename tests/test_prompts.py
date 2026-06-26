from fin_rag.qa.citation import CitationContext, validate_citations
from fin_rag.qa.prompts import REFUSAL_TEMPLATE, build_system_prompt, format_context_block


def _contexts() -> list[CitationContext]:
    return [
        CitationContext(
            index=1,
            domain="fin-online",
            file_path="docs/plan/a.md",
            heading_path="2.3",
            text="账期切换见产品 300007。",
        ),
        CitationContext(
            index=2,
            domain="opdata",
            file_path="docs/channels/b.md",
            heading_path="1",
            text="LanChen 接口说明。",
        ),
    ]


def test_format_context_block_uses_expected_label():
    block = format_context_block(_contexts())

    assert "[1] [fin-online|docs/plan/a.md|§2.3]" in block
    assert "账期切换见产品 300007。" in block


def test_build_system_prompt_contains_refusal_template():
    prompt = build_system_prompt("fin-online", _contexts())

    assert "你是 fin-online 业务知识助手" in prompt
    assert REFUSAL_TEMPLATE in prompt
    assert "<检索上下文>" in prompt


def test_validate_citations_flags_unknown_reference_and_product():
    result = validate_citations("答案见 [3]，产品 300008。", _contexts())

    assert result["valid"] is False
    assert result["invalid_refs"] == [3]
    assert result["suspect_products"] == ["300008"]
