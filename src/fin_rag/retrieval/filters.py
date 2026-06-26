from __future__ import annotations


SUPPORTED_DOMAINS = {"fin-online", "opdata", "shield", "jindun", "hmf", "sjf", "bxf"}


def normalize_domains(domains: list[str] | None) -> list[str]:
    if not domains:
        return []
    normalized: list[str] = []
    for domain in domains:
        lowered = domain.strip().lower()
        if not lowered or lowered == "all":
            continue
        if lowered not in SUPPORTED_DOMAINS:
            raise ValueError(f"Unsupported domain: {domain}")
        if lowered not in normalized:
            normalized.append(lowered)
    return normalized


def build_domain_filter(domains: list[str] | None) -> dict[str, object] | None:
    normalized = normalize_domains(domains)
    if not normalized:
        return None
    return {
        "key": "domain",
        "operator": "in",
        "value": normalized,
    }
