"""Knowledge-base scope validation."""
from t_rag.storage.catalog import validate_id


def validate_knowledge_base(knowledge_base_id: str) -> str:
    return validate_id(knowledge_base_id)
