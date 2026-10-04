from __future__ import annotations

from t_rag.config import Settings


def build_embedding(settings: Settings):
    try:
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index-embeddings-openai-like is not installed."
        ) from exc
    return OpenAILikeEmbedding(
        model_name=settings.embedding_model,
        api_base=settings.embedding_base_url,
        api_key=settings.embedding_api_key or "fake",
        dimensions=settings.embedding_dim,
        embed_batch_size=settings.embedding_batch_size,
    )
