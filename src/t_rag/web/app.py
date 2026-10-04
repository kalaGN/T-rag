from __future__ import annotations

from pathlib import Path
from t_rag.config import Settings
from t_rag.web.management import Management
from t_rag.web.qa import answer


def create_web_app(settings_path: Path | None = None):
    import gradio as gr
    settings = Settings.from_yaml(settings_path)
    management = Management(settings)
    initial_choices = management.choices()
    initial_kb = initial_choices[0][1] if initial_choices else None
    initial_status, initial_sources = management.details(initial_kb)
    with gr.Blocks(title="t-rag") as app:
        gr.Markdown("# t-rag\n本地文档知识库 · 单轮引用问答。远程 LLM 会接收检索出的文本片段。")
        kb = gr.Dropdown(choices=initial_choices, value=initial_kb, label="当前知识库")
        refresh = gr.Button("刷新知识库和状态")
        status = gr.Markdown(initial_status)
        message = gr.Textbox(label="操作结果", interactive=False)
        with gr.Tab("知识库"):
            name = gr.Textbox(label="新知识库名称")
            create = gr.Button("创建知识库")
            confirm = gr.Checkbox(label="确认删除当前知识库及应用管理的数据（目录原始文件保留）")
            delete = gr.Button("删除知识库", variant="stop")
        with gr.Tab("文档与重建"):
            files = gr.File(label="上传文档（同名文件替换）", file_count="multiple", type="filepath",
                            file_types=[".md", ".mdc", ".txt", ".pdf", ".csv", ".xlsx"])
            upload = gr.Button("导入文件")
            directory = gr.Textbox(label="本地目录绝对路径")
            includes = gr.Textbox(label="包含子目录（每行一项，. 表示全部）", value=".")
            excludes = gr.Textbox(label="排除规则（每行一项）", value=".git/**\n.DS_Store")
            add_directory = gr.Button("添加或更新目录")
            sources = gr.Dropdown(choices=initial_sources, value=None, label="已登记的数据源")
            remove = gr.Button("移除所选数据源")
            gr.Markdown("来源变更后请手动重建。重建期间暂停查询；失败后可再次点击重建。仅支持一个应用进程。")
            build = gr.Button("全量重建索引", variant="primary")
        with gr.Tab("问答"):
            question = gr.Textbox(label="问题", lines=3)
            strategy = gr.Radio(["fusion", "vector", "bm25"], value="fusion", label="检索策略")
            ask = gr.Button("提问", variant="primary")
            response = gr.Markdown()
            reason = gr.Textbox(label="回答状态", interactive=False)
            with gr.Accordion("来源与检索片段", open=False):
                evidence = gr.Dataframe(headers=["引用", "文件", "位置", "向量分数", "BM25分数", "片段"],
                                        datatype=["number", "str", "str", "number", "number", "str"], row_count=0, interactive=False)
                document = gr.Dropdown(label="查看已登记的来源")
                preview_button = gr.Button("预览来源文本")
                preview = gr.Textbox(label="来源全文", lines=15, interactive=False)

        def update(kb_id):
            detail, choices = management.details(kb_id)
            return detail, gr.Dropdown(choices=choices, value=None)

        def clear_question():
            return "", "", [], gr.Dropdown(choices=[], value=None), ""

        kb.change(update, [kb], [status, sources]).then(clear_question, outputs=[response, reason, evidence, document, preview])

        def refresh_all(current):
            choices = management.choices()
            selected = current if current in {value for _, value in choices} else None
            return gr.Dropdown(choices=choices, value=selected)
        refresh.click(refresh_all, [kb], [kb]).then(update, [kb], [status, sources])

        def create_handler(value):
            try:
                current = management.create(value)
                return gr.Dropdown(choices=management.choices(), value=current), "知识库已创建。"
            except (ValueError, RuntimeError, OSError) as exc:
                return gr.skip(), str(exc)
        create.click(create_handler, [name], [kb, message]).then(update, [kb], [status, sources])

        def delete_handler(current, confirmed):
            try:
                management.delete(current, confirmed)
                return gr.Dropdown(choices=management.choices(), value=None), "知识库已删除。", False
            except (ValueError, RuntimeError, OSError) as exc:
                return gr.skip(), str(exc), False
        delete.click(delete_handler, [kb, confirm], [kb, message, confirm]).then(update, [kb], [status, sources])

        def operation(callback):
            def handler(*args):
                try:
                    return callback(*args)
                except (ValueError, RuntimeError, OSError) as exc:
                    return str(exc)
            return handler
        upload.click(operation(management.upload), [kb, files], [message]).then(update, [kb], [status, sources])
        add_directory.click(operation(management.directory), [kb, directory, includes, excludes], [message]).then(update, [kb], [status, sources])
        remove.click(operation(management.remove), [kb, sources], [message]).then(update, [kb], [status, sources])

        def build_handler(current, progress=gr.Progress()):
            try:
                return management.rebuild(current, progress)
            except (ValueError, RuntimeError, OSError) as exc:
                return str(exc)
        build.click(build_handler, [kb], [message]).then(update, [kb], [status, sources])

        def ask_handler(current, query, mode):
            try:
                text, rows, choices, info = answer(settings, current, query, mode)
                return text, rows, gr.Dropdown(choices=choices, value=None), info, ""
            except (ValueError, RuntimeError, OSError) as exc:
                return "", [], gr.Dropdown(choices=[], value=None), str(exc), ""
        ask.click(ask_handler, [kb, question, strategy], [response, evidence, document, reason, preview])
        preview_button.click(operation(management.service.preview), [kb, document], [preview])
    return app.queue(default_concurrency_limit=1)


def serve_web(settings_path: Path | None = None, host: str = "127.0.0.1", port: int = 7860) -> None:
    from t_rag.storage.catalog import application_lock
    settings = Settings.from_yaml(settings_path)
    with application_lock(Path(settings.data_dir).expanduser().resolve()):
        create_web_app(settings_path).launch(server_name=host, server_port=port, show_api=False)
