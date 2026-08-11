"""Deterministic and explainable candidate-to-job matching."""

from collections.abc import Iterable

from app.config import TOP_K_RESULTS
from app.models import CandidateProfile, Job, MatchResult
from app.skill_normalizer import normalize_skills, skills_match
from app.vector_store import JobSearchHit


MANDATORY_WEIGHT = 0.50
OPTIONAL_WEIGHT = 0.15
EXPERIENCE_WEIGHT = 0.15
SEMANTIC_WEIGHT = 0.20


def _skill_breakdown(
    candidate_skills: set[str], required_skills: list[str]
) -> tuple[list[str], list[str], float]:
    normalized_requirements = normalize_skills(required_skills)
    matched = [
        skill
        for skill in normalized_requirements
        if any(skills_match(candidate_skill, skill) for candidate_skill in candidate_skills)
    ]
    missing = [skill for skill in normalized_requirements if skill not in matched]
    score = 100.0 if not normalized_requirements else 100.0 * len(matched) / len(normalized_requirements)
    return matched, missing, score


def calculate_experience_score(candidate_years: float, job: Job) -> float:
    """Score experience, allowing a proportional penalty outside the job range."""
    minimum = job.min_experience_years
    maximum = job.max_experience_years

    if minimum is None and maximum is None:
        return 100.0
    if minimum is not None and candidate_years < minimum:
        if minimum == 0:
            return 100.0
        return max(0.0, 100.0 * candidate_years / minimum)
    if maximum is not None and candidate_years > maximum:
        if maximum == 0:
            return 0.0
        excess_ratio = (candidate_years - maximum) / maximum
        return max(0.0, 100.0 * (1.0 - excess_ratio))
    return 100.0


def match_candidate_to_job(
    candidate: CandidateProfile,
    job: Job,
    semantic_score: float,
) -> MatchResult:
    """Calculate a transparent weighted match for one retrieved job."""
    if not 0 <= semantic_score <= 1:
        raise ValueError("semantic_score must be between 0 and 1")

    candidate_skills = set(normalize_skills(candidate.skills))
    matched_mandatory, missing_mandatory, mandatory_score = _skill_breakdown(
        candidate_skills, job.mandatory_skills
    )
    matched_optional, missing_optional, optional_score = _skill_breakdown(
        candidate_skills, job.optional_skills
    )
    experience_score = calculate_experience_score(candidate.experience_years, job)

    final_score = (
        mandatory_score * MANDATORY_WEIGHT
        + optional_score * OPTIONAL_WEIGHT
        + experience_score * EXPERIENCE_WEIGHT
        + semantic_score * 100.0 * SEMANTIC_WEIGHT
    )

    return MatchResult(
        job=job,
        semantic_score=semantic_score,
        mandatory_score=round(mandatory_score, 2),
        optional_score=round(optional_score, 2),
        experience_score=round(experience_score, 2),
        final_score=round(final_score, 2),
        matched_mandatory=matched_mandatory,
        missing_mandatory=missing_mandatory,
        matched_optional=matched_optional,
        missing_optional=missing_optional,
    )


def rank_job_matches(
    candidate: CandidateProfile,
    search_hits: Iterable[JobSearchHit],
    *,
    limit: int = TOP_K_RESULTS,
) -> list[MatchResult]:
    """Score retrieved jobs, sort them, and return the strongest matches."""
    if limit <= 0:
        raise ValueError("Result limit must be greater than zero")

    matches = [
        match_candidate_to_job(candidate, hit.job, hit.semantic_score)
        for hit in search_hits
    ]
    matches = [match for match in matches if match.matched_mandatory]
    matches.sort(
        key=lambda match: (match.final_score, match.semantic_score),
        reverse=True,
    )
    return matches[:limit]
