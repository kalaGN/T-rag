import sys
from pathlib import Path
from types import ModuleType

import pytest

from fin_rag.config import Settings


def _install_fake_fastapi(monkeypatch):
    fake_module = ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, title: str):
            self.title = title
            self.routes = {}

        def get(self, path: str):
            def decorator(func):
                self.routes[("GET", path)] = func
                return func

            return decorator

        def post(self, path: str):
            def decorator(func):
                self.routes[("POST", path)] = func
                return func

            return decorator

    fake_module.FastAPI = FastAPI
    fake_module.HTTPException = HTTPException
    fake_module.Body = lambda default=...: default
    monkeypatch.setitem(sys.modules, "fastapi", fake_module)
    return HTTPException


def test_create_app_exposes_health_and_ask(monkeypatch):
    _install_fake_fastapi(monkeypatch)

    monkeypatch.setattr("fin_rag.api.app.Settings.from_yaml", lambda path=None: Settings(app_name="fin-rag-test"))
    monkeypatch.setattr(
        "fin_rag.api.app.ask_question",
        lambda settings, question, domains=None: type(
            "Result",
            (),
            {
                "answer": "账期切换见 [1]",
                "sources": [{"score": 0.9, "text": "账期切换见产品 300007。", "metadata": {"domain": "fin-online"}}],
                "refused": False,
                "validation": {"valid": True, "invalid_refs": [], "suspect_products": []},
            },
        )(),
    )

    from fin_rag.api.app import create_app

    app = create_app()
    health = app.routes[("GET", "/healthz")]()
    answer = app.routes[("POST", "/ask")]({"question": "账期切换逻辑是什么", "domains": ["fin-online"]})

    assert health == {"status": "ok", "app_name": "fin-rag-test"}
    assert answer["answer"] == "账期切换见 [1]"
    assert answer["refused"] is False


def test_create_app_rejects_bad_payload_and_runtime_error(monkeypatch):
    http_exception = _install_fake_fastapi(monkeypatch)

    monkeypatch.setattr("fin_rag.api.app.Settings.from_yaml", lambda path=None: Settings())

    from fin_rag.api.app import create_app

    app = create_app()
    ask_route = app.routes[("POST", "/ask")]

    with pytest.raises(http_exception) as empty_question:
        ask_route({"question": ""})
    assert empty_question.value.status_code == 400

    with pytest.raises(http_exception) as bad_domains:
        ask_route({"question": "账期切换逻辑是什么", "domains": ["fin-online", ""]})
    assert bad_domains.value.status_code == 400

    monkeypatch.setattr("fin_rag.api.app.ask_question", lambda settings, question, domains=None: (_ for _ in ()).throw(RuntimeError("llm down")))

    with pytest.raises(http_exception) as service_error:
        ask_route({"question": "账期切换逻辑是什么"})
    assert service_error.value.status_code == 503
    assert service_error.value.detail == "llm down"


def test_serve_api_runs_uvicorn(monkeypatch):
    fake_uvicorn = ModuleType("uvicorn")
    captured = {}

    def fake_run(app, host: str, port: int):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    fake_uvicorn.run = fake_run
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr("fin_rag.api.app.create_app", lambda path=None: "app-object")

    from fin_rag.api.app import serve_api

    serve_api(Path("config/settings.yaml"), host="0.0.0.0", port=9000)

    assert captured == {"app": "app-object", "host": "0.0.0.0", "port": 9000}
