from __future__ import annotations

from fin_rag.qa.citation import CitationContext


REFUSAL_TEMPLATE = "知识库未覆盖该问题，请补充更具体的产品、渠道、文件或章节线索。"


def build_system_prompt(domain: str, contexts: list[CitationContext]) -> str:
    resolved_domain = domain or "全库"
    context_block = format_context_block(contexts)
    return (
        f"你是 {resolved_domain} 业务知识助手。严格遵循：\n"
        "1. 只基于<检索上下文>作答，不使用训练记忆，不臆测。\n"
        "2. 每条结论后用 [n] 标注引用，对应文末来源；无依据的问题明确回答"
        f'"{REFUSAL_TEMPLATE}"，绝不编造产品号/PID/渠道/账期/逻辑。\n'
        "3. 涉及配置/SQL/接口时，给出文件路径与章节，必要时贴原文片段。\n"
        "4. 业务术语用文档原词；金额/比例/产品号原样引用，不做四舍五入或推断。\n"
        "5. 若多份文档冲突，列出各来源与差异，交由用户判断，不擅自裁决。\n\n"
        "<检索上下文>\n"
        f"{context_block}\n"
        "</检索上下文>"
    )


def format_context_block(contexts: list[CitationContext]) -> str:
    if not contexts:
        return "[无检索上下文]"
    return "\n".join(context.to_prompt_line() for context in contexts)
