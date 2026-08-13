"""End-to-end resume matching orchestration."""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from app.candidate_extractor import extract_candidate_profile
from app.config import TOP_K_RESULTS, TOP_K_RETRIEVAL
from app.document_builder import build_candidate_document
from app.embedding_service import embed_text
from app.matcher import (
    match_candidate_to_job,
    meets_minimum_experience,
    rank_job_matches,
)
from app.models import CandidateProfile, Job, MatchResult
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


@dataclass(frozen=True, slots=True)
class JobSelectionDiagnostic:
    """Explain how one indexed job moved through the resume-matching gates."""

    reference_number: str
    outcome: str
    explanation: str
    job: Job | None = None
    india_passed: bool | None = None
    status_passed: bool | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    experience_passed: bool | None = None
    match: MatchResult | None = None
    final_rank: int | None = None


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


def explain_job_selection(
    prepared: PreparedResume,
    reference_number: str,
    selected_matches: list[MatchResult],
    *,
    retrieval_limit: int = TOP_K_RETRIEVAL,
    open_only: bool = False,
    vector_store: JobVectorStore | None = None,
) -> JobSelectionDiagnostic:
    """Explain why one reference was selected or rejected without reprocessing a CV."""
    reference = reference_number.strip()
    if not reference:
        raise ValueError("Reference Number is required")

    store = vector_store or JobVectorStore()
    job = next(
        (
            indexed_job
            for indexed_job in store.get_jobs()
            if indexed_job.job_id.casefold() == reference.casefold()
        ),
        None,
    )
    if job is None:
        return JobSelectionDiagnostic(
            reference_number=reference,
            outcome="not_found",
            explanation="This Reference Number is not present in the current job index.",
        )

    india_passed = job.geo.strip().casefold() == "india"
    status_passed = not open_only or job.is_open
    common = {
        "reference_number": job.job_id,
        "job": job,
        "india_passed": india_passed,
        "status_passed": status_passed,
    }
    if not india_passed:
        return JobSelectionDiagnostic(
            **common,
            outcome="country_filter",
            explanation=f"This job is excluded because its Geo is {job.geo}, not India.",
        )
    if not status_passed:
        return JobSelectionDiagnostic(
            **common,
            outcome="status_filter",
            explanation="This job is closed and the Open jobs only filter is enabled.",
        )

    hits = store.search_jobs(
        prepared.candidate_embedding,
        limit=retrieval_limit,
        open_only=open_only,
        india_only=True,
    )
    semantic_rank = next(
        (position for position, hit in enumerate(hits, start=1) if hit.job.job_id == job.job_id),
        None,
    )
    if semantic_rank is None:
        return JobSelectionDiagnostic(
            **common,
            outcome="semantic_retrieval",
            explanation=(
                f"This job was not among the {retrieval_limit} most semantically relevant "
                "eligible jobs for this resume."
            ),
        )

    hit = hits[semantic_rank - 1]
    match = match_candidate_to_job(prepared.candidate, job, hit.semantic_score)
    experience_passed = meets_minimum_experience(prepared.candidate, job)
    details = {
        **common,
        "semantic_rank": semantic_rank,
        "semantic_score": hit.semantic_score,
        "experience_passed": experience_passed,
        "match": match,
    }
    if not experience_passed:
        minimum = job.min_experience_years
        return JobSelectionDiagnostic(
            **details,
            outcome="minimum_experience",
            explanation=(
                f"The candidate has {prepared.candidate.experience_years:g} years of "
                f"experience, below this job's {minimum:g}-year minimum."
            ),
        )
    if not match.matched_mandatory:
        return JobSelectionDiagnostic(
            **details,
            outcome="mandatory_skills",
            explanation="None of this job's mandatory skills matched the candidate evidence.",
        )

    ranked = rank_job_matches(prepared.candidate, hits, limit=retrieval_limit)
    final_rank = next(
        (position for position, ranked_match in enumerate(ranked, start=1) if ranked_match.job.job_id == job.job_id),
        None,
    )
    selected_rank = next(
        (position for position, selected in enumerate(selected_matches, start=1) if selected.job.job_id == job.job_id),
        None,
    )
    if selected_rank is not None:
        return JobSelectionDiagnostic(
            **details,
            final_rank=selected_rank,
            outcome="selected",
            explanation=f"This job passed every gate and was selected at rank {selected_rank}.",
        )
    return JobSelectionDiagnostic(
        **details,
        final_rank=final_rank,
        outcome="below_top_results",
        explanation=(
            f"This job passed every eligibility gate but ranked {final_rank}, below the "
            f"{len(selected_matches)} displayed results."
        ),
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
