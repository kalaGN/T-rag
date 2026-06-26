import pytest

from fin_rag.retrieval.filters import build_domain_filter, normalize_domains


def test_normalize_domains_deduplicates_and_skips_all():
    normalized = normalize_domains(["fin-online", "ALL", "opdata", "fin-online"])

    assert normalized == ["fin-online", "opdata"]


def test_normalize_domains_rejects_unknown_domain():
    with pytest.raises(ValueError):
        normalize_domains(["unknown"])


def test_build_domain_filter_returns_none_for_empty():
    assert build_domain_filter(None) is None
    assert build_domain_filter(["all"]) is None
