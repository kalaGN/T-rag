from __future__ import annotations

import re
from dataclasses import dataclass


PRODUCT_CODE_PATTERN = re.compile(r"\b([36]\d{5})\b")
REFERENCE_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(slots=True)
class CitationContext:
    index: int
    domain: str
    file_path: str
    heading_path: str
    text: str

    def label(self) -> str:
        return f"[{self.domain}|{self.file_path}|§{self.heading_path}]"

    def to_prompt_line(self) -> str:
        return f"[{self.index}] {self.label()} {self.text}"


def extract_reference_numbers(answer: str) -> list[int]:
    return [int(match.group(1)) for match in REFERENCE_PATTERN.finditer(answer)]


def find_suspect_product_codes(answer: str, contexts: list[CitationContext]) -> list[str]:
    allowed_codes = {
        code
        for context in contexts
        for code in PRODUCT_CODE_PATTERN.findall(context.text)
    }
    suspect_codes = []
    for code in PRODUCT_CODE_PATTERN.findall(answer):
        if code not in allowed_codes and code not in suspect_codes:
            suspect_codes.append(code)
    return suspect_codes


def validate_citations(answer: str, contexts: list[CitationContext]) -> dict[str, object]:
    valid_indexes = {context.index for context in contexts}
    invalid_refs = [number for number in extract_reference_numbers(answer) if number not in valid_indexes]
    suspect_products = find_suspect_product_codes(answer, contexts)
    return {
        "valid": not invalid_refs and not suspect_products,
        "invalid_refs": invalid_refs,
        "suspect_products": suspect_products,
    }
