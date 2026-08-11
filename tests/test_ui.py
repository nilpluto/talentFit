"""Tests for Streamlit UI helpers."""

import csv
from io import StringIO
from pathlib import Path

from app.models import CandidateProfile, Job, MatchResult
from app.resume_matching_service import ResumeMatchSummary
from app.ui import build_job_dashboard_dataframe, build_match_report_csv, persist_upload


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
                created_date="11-Aug-2026",
                open_positions=2,
                designation="Senior Engineer",
                geo="India",
                business_unit="Cloud",
                mandatory_skills=["aws", "kubernetes"],
                min_experience_years=4,
                max_experience_years=8,
                status="open",
                referral_allowed=True,
            )
        ]
    )

    assert list(dataframe.columns) == [
        "Reference Number",
        "Job Created Date",
        "Job Title",
        "Open Positions",
        "Designation",
        "Geo",
        "Business Unit",
        "Min Experience",
        "Max Experience",
        "Mandatory Skills",
        "Job Status",
        "Referral Enabled",
    ]
    assert dataframe.iloc[0]["Reference Number"] == "JOB-100"
    assert dataframe.iloc[0]["Referral Enabled"] == True
