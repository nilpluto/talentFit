"""Tests for semantic-search job document generation."""

import pytest

from app.document_builder import (
    build_job_document,
    build_job_documents,
    build_candidate_document,
    format_experience,
)
from app.models import CandidateProfile, Job


def test_build_complete_job_document() -> None:
    job = Job(
        job_id="JOB-001",
        title="Java Backend Engineer",
        designation="Senior Engineer",
        geo="India",
        business_unit="FinTech",
        mandatory_skills=["java", "spring boot", "kafka"],
        min_experience_years=4,
        max_experience_years=7,
        status="open",
    )

    assert build_job_document(job) == "\n".join(
        [
            "Reference Number: JOB-001",
            "Job Title: Java Backend Engineer",
            "Designation: Senior Engineer",
            "Geo: India",
            "Business Unit: FinTech",
            "Mandatory Skills: java, spring boot, kafka",
            "Experience: 4 to 7 years",
            "Job Status: open",
        ]
    )


def test_build_document_handles_missing_optional_values() -> None:
    job = Job(job_id="JOB-002", title="Python Developer")

    document = build_job_document(job)

    assert "Mandatory Skills: None specified" in document
    assert "Experience: Not specified" in document
    assert "Geo: India" in document
    assert "Business Unit: Not specified" in document


@pytest.mark.parametrize(
    ("minimum", "maximum", "expected"),
    [
        (None, None, "Not specified"),
        (None, 5, "Up to 5 years"),
        (4, None, "4+ years"),
        (5, 5, "5 years"),
        (4, 7, "4 to 7 years"),
    ],
)
def test_format_experience(
    minimum: float | None, maximum: float | None, expected: str
) -> None:
    job = Job(
        job_id="JOB-003",
        title="Engineer",
        min_experience_years=minimum,
        max_experience_years=maximum,
    )

    assert format_experience(job) == expected


def test_build_job_documents_preserves_input_order() -> None:
    jobs = [
        Job(job_id="1", title="First Job"),
        Job(job_id="2", title="Second Job"),
    ]

    documents = build_job_documents(jobs)

    assert "Job Title: First Job" in documents[0]
    assert "Job Title: Second Job" in documents[1]


def test_build_candidate_document() -> None:
    candidate = CandidateProfile(
        name="Test Candidate",
        experience_years=5,
        skills=["java", "spring boot", "aws"],
        roles=["backend engineer"],
        location="Bengaluru",
        summary="Backend engineer building Java services.",
    )

    assert build_candidate_document(candidate) == "\n".join(
        [
            "Candidate Roles: backend engineer",
            "Candidate Skills: java, spring boot, aws",
            "Experience: 5 years",
            "Location: Bengaluru",
            "Summary: Backend engineer building Java services.",
        ]
    )
