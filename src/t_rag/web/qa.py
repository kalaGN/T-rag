from t_rag.qa.workflow import ask_question


def answer(settings, kb_id, question, strategy):
    result = ask_question(settings, question, kb_id, strategy=strategy)
    rows = [[source["index"], source["metadata"]["file_path"], source["metadata"]["location"],
             source["scores"].get("vector"), source["scores"].get("bm25"), source["text"]]
            for source in result.sources]
    documents = list(dict.fromkeys((source["metadata"]["file_path"], source["metadata"]["document_id"])
                                  for source in result.sources))
    reason = result.validation.get("refusal_reason")
    return result.answer, rows, documents, (f"证据不足或引用不完整：{reason}" if result.refused else "回答已通过引用编号校验；请结合原文核对内容。")
