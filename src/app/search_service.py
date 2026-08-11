"""Retrieve semantically relevant jobs for a candidate."""

from collections.abc import Callable

from app.config import TOP_K_RETRIEVAL
from app.document_builder import build_candidate_document
from app.embedding_service import embed_text
from app.models import CandidateProfile
from app.vector_store import JobSearchHit, JobVectorStore


CandidateEmbedder = Callable[[str], list[float]]


def search_candidate_jobs(
    candidate: CandidateProfile,
    *,
    vector_store: JobVectorStore | None = None,
    embedder: CandidateEmbedder = embed_text,
    limit: int = TOP_K_RETRIEVAL,
    open_only: bool = False,
    referral_only: bool = False,
) -> list[JobSearchHit]:
    """Embed a candidate and retrieve the nearest indexed jobs."""
    candidate_document = build_candidate_document(candidate)
    candidate_embedding = embedder(candidate_document)
    store = vector_store or JobVectorStore()
    return store.search_jobs(
        candidate_embedding,
        limit=limit,
        open_only=open_only,
        referral_only=referral_only,
    )
