"""End-to-end resume matching orchestration."""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from app.candidate_extractor import extract_candidate_profile
from app.config import TOP_K_RESULTS, TOP_K_RETRIEVAL
from app.document_builder import build_candidate_document
from app.embedding_service import embed_text
from app.matcher import rank_job_matches
from app.models import CandidateProfile, MatchResult
from app.resume_service import extract_resume_text
from app.vector_store import JobVectorStore


@dataclass(frozen=True, slots=True)
class PreparedResume:
    """Candidate data and embedding reusable across different job filters."""

    candidate: CandidateProfile
    candidate_embedding: list[float]
    preparation_timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResumeMatchSummary:
    """Candidate profile and ranked results produced from a resume."""

    candidate: CandidateProfile
    matches: list[MatchResult]
    timings: dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False


def prepare_resume(file_path: str | Path) -> PreparedResume:
    """Extract and embed a resume once for reuse across searches."""
    started = perf_counter()
    resume_text = extract_resume_text(file_path)
    after_pdf = perf_counter()
    candidate = extract_candidate_profile(resume_text)
    after_profile = perf_counter()
    candidate_embedding = embed_text(build_candidate_document(candidate))
    after_embedding = perf_counter()
    return PreparedResume(
        candidate=candidate,
        candidate_embedding=candidate_embedding,
        preparation_timings={
            "pdf_text": after_pdf - started,
            "candidate_extraction": after_profile - after_pdf,
            "candidate_embedding": after_embedding - after_profile,
        },
    )


def match_prepared_resume(
    prepared: PreparedResume,
    *,
    retrieval_limit: int = TOP_K_RETRIEVAL,
    result_limit: int = TOP_K_RESULTS,
    open_only: bool = False,
    cache_hit: bool = False,
    vector_store: JobVectorStore | None = None,
) -> ResumeMatchSummary:
    """Search and rank jobs without repeating resume extraction or embedding."""
    if retrieval_limit <= 0:
        raise ValueError("Retrieval limit must be greater than zero")
    if result_limit <= 0:
        raise ValueError("Result limit must be greater than zero")

    store = vector_store or JobVectorStore()
    search_started = perf_counter()
    search_hits = store.search_jobs(
        prepared.candidate_embedding,
        limit=retrieval_limit,
        open_only=open_only,
        india_only=True,
    )
    after_search = perf_counter()
    matches = rank_job_matches(prepared.candidate, search_hits, limit=result_limit)
    after_matching = perf_counter()
    timings = (
        {key: 0.0 for key in prepared.preparation_timings}
        if cache_hit
        else dict(prepared.preparation_timings)
    )
    timings.update(
        {
            "vector_search": after_search - search_started,
            "matching": after_matching - after_search,
        }
    )
    return ResumeMatchSummary(
        candidate=prepared.candidate,
        matches=matches,
        timings=timings,
        cache_hit=cache_hit,
    )


def match_resume(
    file_path: str | Path,
    *,
    retrieval_limit: int = TOP_K_RETRIEVAL,
    result_limit: int = TOP_K_RESULTS,
    open_only: bool = False,
) -> ResumeMatchSummary:
    """Extract a resume and return its strongest explainable job matches."""
    if retrieval_limit <= 0:
        raise ValueError("Retrieval limit must be greater than zero")
    if result_limit <= 0:
        raise ValueError("Result limit must be greater than zero")
    prepared = prepare_resume(file_path)
    return match_prepared_resume(
        prepared,
        retrieval_limit=retrieval_limit,
        result_limit=result_limit,
        open_only=open_only,
    )
