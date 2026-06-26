from fin_rag.embedding_service import (
    build_embeddings_response,
    build_models_response,
    estimate_token_count,
    normalize_inputs,
)


def test_normalize_inputs_accepts_string_and_list():
    assert normalize_inputs("abc") == ["abc"]
    assert normalize_inputs(["a", "b"]) == ["a", "b"]


def test_build_models_response_shape():
    payload = build_models_response("BAAI/bge-base-zh-v1.5")

    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "BAAI/bge-base-zh-v1.5"


def test_build_embeddings_response_shape():
    payload = build_embeddings_response(
        model="BAAI/bge-base-zh-v1.5",
        texts=["账期切换"],
        vectors=[[0.1, 0.2]],
    )

    assert payload["model"] == "BAAI/bge-base-zh-v1.5"
    assert payload["data"][0]["embedding"] == [0.1, 0.2]
    assert payload["usage"]["prompt_tokens"] == estimate_token_count("账期切换")
