"""Tests for Streamlit UI helpers."""

import csv
from io import StringIO
from pathlib import Path

from app.models import CandidateProfile, Job, MatchResult
from app.resume_matching_service import ResumeMatchSummary
from app.ui import (
    build_job_dashboard_dataframe,
    build_match_report_csv,
    filter_jobs_by_reference,
    persist_upload,
)


class FakeUpload:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_persist_upload_uses_safe_name_and_preserves_content(tmp_path: Path) -> None:
    upload = FakeUpload("../../candidate.PDF", b"sample PDF bytes")

    path = persist_upload(upload, tmp_path)

    assert path == tmp_path / "upload.pdf"
    assert path.read_bytes() == b"sample PDF bytes"


def test_build_match_report_csv_contains_candidate_and_score_details() -> None:
    job = Job(
        job_id="JOB-001",
        title="Java Backend Engineer",
        mandatory_skills=["java", "spring boot"],
        location="Bengaluru",
        status="open",
        min_experience_years=4,
        max_experience_years=7,
    )
    summary = ResumeMatchSummary(
        candidate=CandidateProfile(
            name="Test Candidate",
            experience_years=5,
            skills=["java", "spring boot"],
            roles=["backend engineer"],
        ),
        matches=[
            MatchResult(
                job=job,
                semantic_score=0.9,
                mandatory_score=100,
                optional_score=100,
                experience_score=100,
                final_score=98,
                matched_mandatory=["java", "spring boot"],
            )
        ],
    )

    rows = list(csv.DictReader(StringIO(build_match_report_csv(summary).decode())))

    assert len(rows) == 1
    assert rows[0]["candidate_name"] == "Test Candidate"
    assert rows[0]["job_id"] == "JOB-001"
    assert rows[0]["final_score"] == "98.0"
    assert rows[0]["matched_mandatory"] == "java; spring boot"


def test_build_job_dashboard_dataframe_uses_supported_ats_fields() -> None:
    dataframe = build_job_dashboard_dataframe(
        [
            Job(
                job_id="JOB-100",
                title="Platform Engineer",
                designation="Senior Engineer",
                geo="India",
                business_unit="Cloud",
                mandatory_skills=["aws", "kubernetes"],
                min_experience_years=4,
                max_experience_years=8,
                status="open",
            )
        ]
    )

    assert list(dataframe.columns) == [
        "Reference Number",
        "Job Title",
        "Designation",
        "Geo",
        "Business Unit",
        "Min Experience",
        "Max Experience",
        "Mandatory Skills",
        "Job Status",
    ]
    assert dataframe.iloc[0]["Reference Number"] == "JOB-100"


def test_filter_jobs_by_full_or_partial_reference_number() -> None:
    jobs = [
        Job(job_id="REQ-18846", title="First", geo="India"),
        Job(job_id="REQ-20001", title="Second", geo="India"),
        Job(job_id="IND-18847", title="Third", geo="India"),
    ]

    assert filter_jobs_by_reference(jobs, "") == jobs
    assert filter_jobs_by_reference(jobs, "  req-18846  ") == [jobs[0]]
    assert filter_jobs_by_reference(jobs, "REQ") == jobs[:2]
    assert filter_jobs_by_reference(jobs, "1884") == [jobs[0], jobs[2]]
    assert filter_jobs_by_reference(jobs, "missing") == []
