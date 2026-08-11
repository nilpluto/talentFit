"""Tests for Ollama embedding generation."""

import pytest

from app.embedding_service import embed_text, embed_texts


class FakeEmbeddingClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[tuple[str, list[str], str]] = []

    def embed(
        self, *, model: str, input: list[str], keep_alive: str
    ) -> dict[str, list[list[float]]]:
        self.calls.append((model, input, keep_alive))
        return {"embeddings": self.embeddings}


def test_embed_multiple_texts() -> None:
    client = FakeEmbeddingClient([[0.1, 0.2], [0.3, 0.4]])

    embeddings = embed_texts(
        ["Java backend engineer", "Python Django developer"],
        model="test-model",
        client=client,
    )

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert client.calls == [
        ("test-model", ["Java backend engineer", "Python Django developer"], "30m")
    ]


def test_embed_single_text() -> None:
    client = FakeEmbeddingClient([[0.5, 0.6]])

    assert embed_text("Java engineer", client=client) == [0.5, 0.6]


def test_empty_collection_returns_empty_list() -> None:
    assert embed_texts([]) == []


def test_blank_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty text"):
        embed_texts(["Java engineer", "  "])


def test_embedding_count_must_match_input_count() -> None:
    client = FakeEmbeddingClient([[0.1, 0.2]])

    with pytest.raises(RuntimeError, match="1 embeddings for 2 texts"):
        embed_texts(["first", "second"], client=client)
