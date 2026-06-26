from __future__ import annotations

from pathlib import Path

from fin_rag.config import Settings
from fin_rag.qa.workflow import ask_question


def create_app(settings_path: Path | None = None):
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI is not installed. Run ./.venv/bin/pip install -e '.[dev]' first.") from exc

    settings = Settings.from_yaml(settings_path)
    app = FastAPI(title="fin-rag")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "app_name": settings.app_name}

    @app.post("/ask")
    def ask(payload: dict[str, object] = Body(...)) -> dict[str, object]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required.")

        domains = payload.get("domains")
        if domains is not None:
            if not isinstance(domains, list) or any(not isinstance(item, str) or not item.strip() for item in domains):
                raise HTTPException(status_code=400, detail="domains must be a list of non-empty strings.")

        try:
            result = ask_question(settings, question, domains=domains)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return {
            "answer": result.answer,
            "sources": result.sources,
            "refused": result.refused,
            "validation": result.validation,
        }

    return app


def serve_api(
    settings_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8010,
) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError("uvicorn is not installed. Run ./.venv/bin/pip install -e '.[dev]' first.") from exc

    uvicorn.run(create_app(settings_path), host=host, port=port)
