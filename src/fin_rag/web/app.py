from __future__ import annotations

import json
from pathlib import Path

from fin_rag.config import Settings
from fin_rag.qa.workflow import ask_question


def create_web_app(settings_path: Path | None = None):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("gradio is not installed.") from exc

    settings = Settings.from_yaml(settings_path)

    with gr.Blocks(title="fin-rag") as app:
        gr.Markdown("# fin-rag\n最小问答页面，直接复用当前知识库问答链路。")

        domain_selector = gr.CheckboxGroup(
            choices=["fin-online", "opdata"],
            label="业务域过滤",
            value=[],
        )
        question_box = gr.Textbox(label="问题", placeholder="例如：账期切换逻辑是什么")
        submit_button = gr.Button("提问")
        answer_box = gr.Markdown(label="回答")
        sources_box = gr.JSON(label="来源")
        validation_box = gr.JSON(label="校验结果")

        def run(question: str, domains: list[str] | None):
            cleaned_question = question.strip()
            if not cleaned_question:
                return "请输入问题。", [], {"valid": False, "error": "question is required"}
            result = ask_question(settings, cleaned_question, domains=domains or None)
            return result.answer, result.sources, result.validation

        submit_button.click(
            fn=run,
            inputs=[question_box, domain_selector],
            outputs=[answer_box, sources_box, validation_box],
        )

    return app


def serve_web(
    settings_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    app = create_web_app(settings_path)
    app.launch(server_name=host, server_port=port)
