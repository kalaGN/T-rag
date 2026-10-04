from __future__ import annotations

import re
from dataclasses import dataclass

REFERENCE_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(slots=True)
class CitationContext:
    index: int
    domain: str
    file_path: str
    heading_path: str
    text: str

    def label(self) -> str:
        return f"{self.file_path} · {self.heading_path}"

    def to_prompt_line(self) -> str:
        return f"[{self.index}] {self.label()}\n{self.text}"


def extract_reference_numbers(answer: str) -> list[int]:
    return [int(match.group(1)) for match in REFERENCE_PATTERN.finditer(answer)]


def validate_citations(answer: str, contexts: list[CitationContext]) -> dict:
    numbers = extract_reference_numbers(answer)
    valid_indexes = {context.index for context in contexts}
    invalid = sorted(set(number for number in numbers if number not in valid_indexes))
    reason = None
    if not contexts:
        reason = "no_sources"
    elif not answer.strip():
        reason = "empty_answer"
    elif not numbers:
        reason = "missing_refs"
    elif invalid:
        reason = "invalid_refs"
    return {"valid": reason is None, "invalid_refs": invalid, "refusal_reason": reason}
