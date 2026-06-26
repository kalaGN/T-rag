from __future__ import annotations


TERM_SYNONYMS = {
    "账期": "结算周期",
    "降查得": "降档",
    "billing cycle": "结算周期",
}


def normalize_query(query: str) -> str:
    normalized = query
    lowered = query.lower()
    for source, target in TERM_SYNONYMS.items():
        if source in query or source in lowered:
            normalized = f"{normalized}\n{target}"
    return normalized
