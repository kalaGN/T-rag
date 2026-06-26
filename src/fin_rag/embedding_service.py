from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

from fin_rag.config import Settings


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    model_name: str
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install dependencies before starting the embedding service."
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]


def serve(settings: Settings) -> None:
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    server = build_server(settings=settings, embedder=embedder)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def build_server(settings: Settings, embedder: Embedder) -> ThreadingHTTPServer:
    handler = _build_handler(settings=settings, embedder=embedder)
    return ThreadingHTTPServer((settings.embedding_host, settings.embedding_port), handler)


def _build_handler(settings: Settings, embedder: Embedder):
    class EmbeddingHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self._json(404, {"error": "not found"})
                return
            self._json(200, build_models_response(settings.embedding_model))

        def do_POST(self) -> None:
            if self.path != "/v1/embeddings":
                self._json(404, {"error": "not found"})
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")) or 0)
            payload = json.loads(body or b"{}")
            texts = normalize_inputs(payload.get("input"))
            vectors = embedder.encode(texts)
            self._json(200, build_embeddings_response(model=settings.embedding_model, texts=texts, vectors=vectors))

        def log_message(self, format: str, *args) -> None:
            return

        def _json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return EmbeddingHandler


def normalize_inputs(raw_input) -> list[str]:
    if isinstance(raw_input, str):
        return [raw_input]
    if isinstance(raw_input, list) and all(isinstance(item, str) for item in raw_input):
        return raw_input
    raise ValueError("input must be a string or list of strings")


def build_models_response(model_name: str) -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
            }
        ],
    }


def build_embeddings_response(model: str, texts: list[str], vectors: list[list[float]]) -> dict[str, object]:
    data = []
    total_tokens = 0
    for index, (text, vector) in enumerate(zip(texts, vectors)):
        token_count = estimate_token_count(text)
        total_tokens += token_count
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": vector,
            }
        )
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


def estimate_token_count(text: str) -> int:
    return max(1, len(text.strip()))
