from __future__ import annotations

import re
from pathlib import Path

from fin_rag.config import SourceConfig


PRODUCT_CODE_PATTERN = re.compile(r"\b([36]\d{5})\b")
CHANNEL_KEYWORDS = ("cucc", "cmcc", "ctcc", "lanchen", "rong360", "shield", "jindun")


def build_metadata(source: SourceConfig, relative_path: Path, text: str) -> dict[str, str | None]:
    lower_path = relative_path.as_posix().lower()
    product_code = _find_product_code(relative_path.name, text)
    channel = _find_channel(lower_path, text)
    return {
        "project": source.name,
        "domain": _infer_domain(source, lower_path, text),
        "source_type": _infer_source_type(lower_path),
        "doc_type": _infer_doc_type(lower_path),
        "product_code": product_code,
        "channel": channel,
        "file_path": relative_path.as_posix(),
    }


def _infer_domain(source: SourceConfig, lower_path: str, text: str) -> str:
    joined = f"{lower_path}\n{text.lower()}"
    for keyword in ("shield", "jindun", "hmf", "sjf", "bxf"):
        if keyword in joined:
            return keyword
    if "opdata" in joined:
        return "opdata"
    return source.default_domain


def _infer_source_type(lower_path: str) -> str:
    if ".cursor/memory/" in lower_path:
        return "cursor_memory"
    if ".cursor/rules/" in lower_path:
        return "cursor_rule"
    if lower_path.endswith((".java", ".xml", ".py")):
        return "code_sample"
    if "/channels/" in lower_path:
        return "channel_doc"
    return "doc"


def _infer_doc_type(lower_path: str) -> str:
    if "/spec/" in lower_path:
        return "spec"
    if "/plan/" in lower_path:
        return "tech_plan"
    if ".cursor/memory/special/" in lower_path:
        return "business_logic"
    if ".cursor/rules/" in lower_path:
        return "engineering_rule"
    if "/channels/" in lower_path:
        return "channel_doc"
    return "doc"


def _find_product_code(filename: str, text: str) -> str | None:
    for target in (filename, text):
        match = PRODUCT_CODE_PATTERN.search(target)
        if match:
            return match.group(1)
    return None


def _find_channel(lower_path: str, text: str) -> str | None:
    joined = f"{lower_path}\n{text.lower()}"
    for keyword in CHANNEL_KEYWORDS:
        if keyword in joined:
            return keyword
    return None
