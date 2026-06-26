import sys
from pathlib import Path
from types import ModuleType

from fin_rag.config import Settings


def _install_fake_gradio(monkeypatch):
    fake_module = ModuleType("gradio")
    captured = {"clicks": []}

    class Blocks:
        def __init__(self, title: str):
            self.title = title
            self.launch_calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def launch(self, server_name: str, server_port: int):
            self.launch_calls.append({"server_name": server_name, "server_port": server_port})

    class _Component:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Button(_Component):
        def click(self, fn, inputs, outputs):
            captured["clicks"].append({"fn": fn, "inputs": inputs, "outputs": outputs})

    fake_module.Blocks = Blocks
    fake_module.Markdown = _Component
    fake_module.CheckboxGroup = _Component
    fake_module.Textbox = _Component
    fake_module.Button = Button
    fake_module.JSON = _Component
    monkeypatch.setitem(sys.modules, "gradio", fake_module)
    return captured


def test_create_web_app_registers_submit_handler(monkeypatch):
    captured = _install_fake_gradio(monkeypatch)
    monkeypatch.setattr("fin_rag.web.app.Settings.from_yaml", lambda path=None: Settings())
    monkeypatch.setattr(
        "fin_rag.web.app.ask_question",
        lambda settings, question, domains=None: type(
            "Result",
            (),
            {
                "answer": "账期切换见 [1]",
                "sources": [{"score": 0.9, "text": "账期切换见产品 300007。", "metadata": {"domain": "fin-online"}}],
                "validation": {"valid": True, "invalid_refs": [], "suspect_products": []},
            },
        )(),
    )

    from fin_rag.web.app import create_web_app

    app = create_web_app()
    handler = captured["clicks"][0]["fn"]
    answer, sources, validation = handler("账期切换逻辑是什么", ["fin-online"])

    assert app.title == "fin-rag"
    assert answer == "账期切换见 [1]"
    assert sources[0]["metadata"]["domain"] == "fin-online"
    assert validation["valid"] is True


def test_create_web_app_rejects_empty_question(monkeypatch):
    captured = _install_fake_gradio(monkeypatch)
    monkeypatch.setattr("fin_rag.web.app.Settings.from_yaml", lambda path=None: Settings())

    from fin_rag.web.app import create_web_app

    create_web_app()
    handler = captured["clicks"][0]["fn"]
    answer, sources, validation = handler("   ", [])

    assert answer == "请输入问题。"
    assert sources == []
    assert validation["valid"] is False


def test_serve_web_launches_blocks(monkeypatch):
    _install_fake_gradio(monkeypatch)
    fake_app = type(
        "FakeApp",
        (),
        {"launch": lambda self, server_name, server_port: setattr(self, "launch_args", (server_name, server_port))},
    )()
    monkeypatch.setattr("fin_rag.web.app.create_web_app", lambda path=None: fake_app)

    from fin_rag.web.app import serve_web

    serve_web(Path("config/settings.yaml"), host="0.0.0.0", port=7861)

    assert fake_app.launch_args == ("0.0.0.0", 7861)
