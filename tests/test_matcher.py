"""Tests for deterministic candidate-to-job matching."""

import pytest

from app.matcher import (
    calculate_experience_score,
    match_candidate_to_job,
    rank_job_matches,
)
from app.models import CandidateProfile, Job
from app.vector_store import JobSearchHit


@pytest.fixture
def candidate() -> CandidateProfile:
    return CandidateProfile(
        name="Test Candidate",
        experience_years=5,
        skills=["java", "spring boot", "aws"],
        roles=["backend engineer"],
    )


def test_match_has_weighted_score_and_skill_gaps(candidate: CandidateProfile) -> None:
    job = Job(
        job_id="JOB-001",
        title="Java Backend Engineer",
        mandatory_skills=["java", "spring boot", "kafka"],
        optional_skills=["aws", "kubernetes"],
        min_experience_years=4,
        max_experience_years=7,
    )

    result = match_candidate_to_job(candidate, job, semantic_score=0.8)

    assert result.matched_mandatory == ["java", "spring boot"]
    assert result.missing_mandatory == ["kafka"]
    assert result.matched_optional == ["aws"]
    assert result.missing_optional == ["kubernetes"]
    assert result.mandatory_score == 66.67
    assert result.optional_score == 50
    assert result.experience_score == 100
    assert result.final_score == 71.83


@pytest.mark.parametrize(
    ("candidate_years", "minimum", "maximum", "expected"),
    [
        (5, 4, 7, 100),
        (3, 4, 7, 75),
        (8, 4, 7, pytest.approx(85.7142857)),
        (5, None, None, 100),
    ],
)
def test_experience_scoring(
    candidate_years: float,
    minimum: float | None,
    maximum: float | None,
    expected: float,
) -> None:
    job = Job(
        job_id="JOB",
        title="Engineer",
        min_experience_years=minimum,
        max_experience_years=maximum,
    )

    assert calculate_experience_score(candidate_years, job) == expected


def test_jobs_without_skill_requirements_do_not_penalize_candidate(
    candidate: CandidateProfile,
) -> None:
    result = match_candidate_to_job(
        candidate,
        Job(job_id="JOB", title="Engineer"),
        semantic_score=1,
    )

    assert result.mandatory_score == 100
    assert result.optional_score == 100
    assert result.final_score == 100


def test_rank_returns_top_three_by_final_score(candidate: CandidateProfile) -> None:
    hits = [
        JobSearchHit(
            job=Job(
                job_id=f"JOB-{index}",
                title=f"Job {index}",
                mandatory_skills=skills,
                min_experience_years=4,
                max_experience_years=7,
            ),
            document="document",
            distance=1 - semantic,
            semantic_score=semantic,
        )
        for index, skills, semantic in [
            (1, ["java", "spring boot"], 0.9),
            (2, ["java"], 0.8),
            (3, ["python"], 0.95),
            (4, ["go"], 0.5),
        ]
    ]

    matches = rank_job_matches(candidate, hits, limit=3)

    assert [match.job.job_id for match in matches] == ["JOB-1", "JOB-2", "JOB-3"]


def test_reject_invalid_semantic_score(candidate: CandidateProfile) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        match_candidate_to_job(
            candidate,
            Job(job_id="JOB", title="Engineer"),
            semantic_score=1.1,
        )
