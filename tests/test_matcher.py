"""Tests for deterministic candidate-to-job matching."""

import pytest

from app.matcher import (
    calculate_experience_score,
    match_candidate_to_job,
    meets_minimum_experience,
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


def test_match_uses_canonical_variants_for_mandatory_skills() -> None:
    candidate = CandidateProfile(
        name="BI Candidate",
        experience_years=5,
        skills=["PowerBI", "Service Now", "React.js"],
    )
    job = Job(
        job_id="JOB-VARIANTS",
        title="Application Developer",
        mandatory_skills=["Power BI Desktop", "ServiceNow", "React JS"],
    )

    result = match_candidate_to_job(candidate, job, semantic_score=0.7)

    assert result.matched_mandatory == ["power bi", "servicenow", "react"]
    assert result.missing_mandatory == []
    assert result.mandatory_score == 100


def test_professional_role_can_evidence_an_equivalent_mandatory_skill() -> None:
    candidate = CandidateProfile(
        name="Reliability Candidate",
        experience_years=8,
        skills=["kubernetes", "aws"],
        roles=["Senior Site Reliability Engineer"],
    )
    job = Job(
        job_id="JOB-SRE",
        title="Site Reliability Engineer",
        mandatory_skills=["sre"],
        min_experience_years=7,
    )

    result = match_candidate_to_job(candidate, job, semantic_score=0.8)

    assert result.matched_mandatory == ["sre"]
    assert result.missing_mandatory == []


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


@pytest.mark.parametrize(
    ("candidate_years", "minimum", "expected"),
    [
        (3, 4, False),
        (4, 4, True),
        (5, 4, True),
        (0, None, True),
    ],
)
def test_minimum_experience_gate(
    candidate_years: float, minimum: float | None, expected: bool
) -> None:
    candidate = CandidateProfile(name="Candidate", experience_years=candidate_years)
    job = Job(
        job_id="JOB-MINIMUM",
        title="Engineer",
        min_experience_years=minimum,
    )

    assert meets_minimum_experience(candidate, job) is expected


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


def test_rank_excludes_jobs_without_a_mandatory_skill_match(
    candidate: CandidateProfile,
) -> None:
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

    assert [match.job.job_id for match in matches] == ["JOB-1", "JOB-2"]


def test_rank_returns_no_jobs_when_no_mandatory_skills_match(
    candidate: CandidateProfile,
) -> None:
    hits = [
        JobSearchHit(
            job=Job(
                job_id="JOB-PYTHON",
                title="Python Engineer",
                mandatory_skills=["python", "django"],
            ),
            document="document",
            distance=0.01,
            semantic_score=0.99,
        )
    ]

    assert rank_job_matches(candidate, hits, limit=3) == []


def test_rank_excludes_job_before_scoring_when_candidate_is_below_minimum() -> None:
    candidate = CandidateProfile(
        name="Junior Candidate",
        experience_years=3,
        skills=["java", "spring boot"],
    )
    hit = JobSearchHit(
        job=Job(
            job_id="JOB-SENIOR",
            title="Senior Java Engineer",
            mandatory_skills=["java", "spring boot"],
            min_experience_years=4,
        ),
        document="document",
        distance=0,
        semantic_score=1,
    )

    assert rank_job_matches(candidate, [hit], limit=3) == []


def test_rank_includes_job_when_candidate_meets_minimum_exactly() -> None:
    candidate = CandidateProfile(
        name="Qualified Candidate",
        experience_years=4,
        skills=["java"],
    )
    hit = JobSearchHit(
        job=Job(
            job_id="JOB-QUALIFIED",
            title="Java Engineer",
            mandatory_skills=["java"],
            min_experience_years=4,
        ),
        document="document",
        distance=0.1,
        semantic_score=0.9,
    )

    assert [match.job.job_id for match in rank_job_matches(candidate, [hit])] == [
        "JOB-QUALIFIED"
    ]


def test_reject_invalid_semantic_score(candidate: CandidateProfile) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        match_candidate_to_job(
            candidate,
            Job(job_id="JOB", title="Engineer"),
            semantic_score=1.1,
        )
