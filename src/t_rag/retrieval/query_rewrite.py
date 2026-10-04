def normalize_query(query: str) -> str:
    """Only normalize whitespace; no domain-specific expansion."""
    return query.strip()
