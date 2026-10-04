import pytest
from t_rag.retrieval.filters import validate_knowledge_base


def test_scope_requires_internal_id():
    assert validate_knowledge_base("a" * 32) == "a" * 32
    for value in ("../other", "", None, "fin-online"):
        with pytest.raises(ValueError):
            validate_knowledge_base(value)
