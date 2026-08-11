"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")

    return value


CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "data" / "chroma")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "talentfit_jobs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
TOP_K_RETRIEVAL = _positive_int("TOP_K_RETRIEVAL", 10)
TOP_K_RESULTS = _positive_int("TOP_K_RESULTS", 3)
