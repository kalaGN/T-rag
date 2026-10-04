from __future__ import annotations

from dataclasses import dataclass
from t_rag.config import Settings
from t_rag.models.llm import build_llm
from t_rag.qa.citation import CitationContext, validate_citations
from t_rag.qa.prompts import REFUSAL_TEMPLATE, build_system_prompt
from t_rag.retrieval.fusion import retrieve
from t_rag.storage.catalog import LOCK


@dataclass(slots=True)
class QAResult:
    answer: str
    sources: list[dict]
    refused: bool
    validation: dict
    raw_answer: str


def ask_question(settings: Settings, question: str, knowledge_base_id: str, llm=None, strategy="fusion") -> QAResult:
    # Keep the index stable throughout retrieval and answer generation.
    with LOCK:
        hits = retrieve(settings, knowledge_base_id, question, strategy)
        contexts = [CitationContext(i, knowledge_base_id, hit.metadata["file_path"], hit.metadata["location"], hit.text)
                    for i, hit in enumerate(hits, 1)]
        answer = ""
        if contexts:
            active_llm = llm or build_llm(settings)
            try:
                response = active_llm.complete(build_system_prompt(knowledge_base_id, contexts) + f"\n\n<问题>{question}</问题>")
            except Exception as exc:
                raise RuntimeError("回答生成失败，请检查 LLM 配置、服务可用性和超时设置。") from exc
            answer = (response.text or "").strip()
        validation = validate_citations(answer, contexts)
        if REFUSAL_TEMPLATE in answer:
            validation.update(valid=False, refusal_reason="insufficient_evidence")
        refused = not validation["valid"]
        sources = [{"index": i, **hit.as_dict()} for i, hit in enumerate(hits, 1)]
        return QAResult(REFUSAL_TEMPLATE if refused else answer, sources, refused, validation, answer)
