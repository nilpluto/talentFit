"""Generate semantic embeddings through the local Ollama service."""

from collections.abc import Sequence
from typing import Protocol

from ollama import Client

from app.config import EMBEDDING_MODEL, OLLAMA_HOST


class EmbeddingClient(Protocol):
    """Minimal Ollama client interface used by this service."""

    def embed(self, *, model: str, input: Sequence[str]) -> object: ...


def _extract_embeddings(response: object) -> list[list[float]]:
    if hasattr(response, "embeddings"):
        embeddings = response.embeddings
    elif isinstance(response, dict):
        embeddings = response.get("embeddings")
    else:
        embeddings = None

    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings")

    return [[float(value) for value in vector] for vector in embeddings]


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = EMBEDDING_MODEL,
    host: str = OLLAMA_HOST,
    client: EmbeddingClient | None = None,
) -> list[list[float]]:
    """Embed one or more non-empty texts using the configured Ollama model."""
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Embedding input must not contain empty text")

    ollama_client = client or Client(host=host)
    response = ollama_client.embed(
        model=model,
        input=list(texts),
        keep_alive="30m",
    )
    embeddings = _extract_embeddings(response)

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
        )

    return embeddings


def embed_text(
    text: str,
    *,
    model: str = EMBEDDING_MODEL,
    host: str = OLLAMA_HOST,
    client: EmbeddingClient | None = None,
) -> list[float]:
    """Embed a single text using the configured Ollama model."""
    return embed_texts([text], model=model, host=host, client=client)[0]
