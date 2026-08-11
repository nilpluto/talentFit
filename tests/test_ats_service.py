"""Tests for ATS file loading."""

from pathlib import Path

import pandas as pd
import pytest

from app.ats_service import (
    clean_description,
    dataframe_to_jobs,
    load_jobs,
    normalize_job_status,
    normalize_referral_flag,
    parse_experience_range,
    read_ats_file,
)


SAMPLE_JOBS = Path(__file__).parents[1] / "resources" / "sample_ats_1.xlsx"


def test_load_sample_csv_as_jobs() -> None:
    jobs = load_jobs(SAMPLE_JOBS)

    assert len(jobs) == 4
    assert jobs[0].job_id == "18846"
    assert jobs[0].title == "Power BI Developer"
    assert jobs[0].created_date == "02-Jan-2026"
    assert jobs[0].open_positions == 1
    assert jobs[0].designation == "Senior Lead Data Analyst"
    assert jobs[0].geo == "India"
    assert jobs[0].business_unit == "FinTech"
    assert jobs[0].mandatory_skills[:2] == ["power bi", "power bi reports"]
    assert jobs[0].min_experience_years == 5
    assert jobs[0].max_experience_years is None
    assert jobs[0].referral_allowed is False
    assert jobs[0].status == "rejected"
    assert jobs[0].description == ""


def test_load_excel_as_jobs(tmp_path: Path) -> None:
    excel_path = tmp_path / "jobs.xlsx"
    pd.DataFrame(
        [{"job_id": "JOB-X", "title": "Test Engineer", "mandatory_skills": "Python"}]
    ).to_excel(excel_path, index=False)

    jobs = load_jobs(excel_path)

    assert len(jobs) == 1
    assert jobs[0].job_id == "JOB-X"
    assert jobs[0].mandatory_skills == ["python"]


def test_reject_file_missing_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame([{"title": "Missing ID"}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="reference_number"):
        load_jobs(csv_path)


def test_reject_unsupported_file_type(tmp_path: Path) -> None:
    json_path = tmp_path / "jobs.json"
    json_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported ATS file type"):
        read_ats_file(json_path)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("4-7 years", (4.0, 7.0)),
        ("5+ years", (5.0, None)),
        ("3 years", (3.0, 3.0)),
        (None, (None, None)),
        ("Not specified", (None, None)),
    ],
)
def test_parse_experience_range(
    raw_value: object, expected: tuple[float | None, float | None]
) -> None:
    assert parse_experience_range(raw_value) == expected


def test_clean_description_removes_html_and_extra_whitespace() -> None:
    description = "<p>Build <strong>Java</strong> APIs.</p>\n<p>Work with Kafka.</p>"

    assert clean_description(description) == "Build Java APIs. Work with Kafka."
    assert clean_description(None) == ""


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("ACTIVE", "open"),
        ("Published", "open"),
        ("Filled", "closed"),
        ("Paused", "on hold"),
        (None, "unknown"),
    ],
)
def test_normalize_job_status(raw_status: object, expected: str) -> None:
    assert normalize_job_status(raw_status) == expected


@pytest.mark.parametrize(
    ("raw_flag", "expected"),
    [(True, True), ("YES", True), (1, True), (False, False), ("no", False), (None, False)],
)
def test_normalize_referral_flag(raw_flag: object, expected: bool) -> None:
    assert normalize_referral_flag(raw_flag) is expected


def test_header_aliases_extra_columns_and_missing_values_become_clean_job() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "job_id": "JOB-100",
                "title": "Platform Engineer",
                "Min Exp": 4,
                "Max Exp": 7,
                "mandatory_skills": "AWS; K8s; ",
                "Geo": None,
                "Job Status": "Active",
                "Refferal Enabled": "Y",
                "Unrelated Export Column": "ignored",
            }
        ]
    )

    job = dataframe_to_jobs(dataframe)[0]

    assert job.min_experience_years == 4
    assert job.max_experience_years == 7
    assert job.mandatory_skills == ["aws", "kubernetes"]
    assert job.optional_skills == []
    assert job.description == ""
    assert job.geo is None
    assert job.status == "open"
    assert job.referral_allowed is True
