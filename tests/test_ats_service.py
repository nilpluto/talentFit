"""Tests for ATS file loading."""

from pathlib import Path

import pandas as pd
import pytest

from app.ats_service import (
    clean_description,
    dataframe_to_jobs,
    load_jobs,
    normalize_job_status,
    parse_experience_columns,
    parse_experience_range,
    read_ats_file,
)


SAMPLE_JOBS = Path(__file__).parents[1] / "resources" / "sample_ats_1.xlsx"


def test_load_sample_csv_as_jobs() -> None:
    jobs = load_jobs(SAMPLE_JOBS)

    assert len(jobs) == 4
    assert jobs[0].job_id == "18846"
    assert jobs[0].title == "Power BI Developer"
    assert jobs[0].designation == "Senior Lead Data Analyst"
    assert jobs[0].geo == "India"
    assert jobs[0].business_unit == "FinTech"
    assert jobs[0].mandatory_skills == [
        "power bi",
        "business intelligence developer",
        "power query",
    ]
    assert jobs[0].min_experience_years == 5
    assert jobs[0].max_experience_years is None
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
        ("6 Months", (0.5, 0.5)),
        ("6-18 months", (0.5, 1.5)),
        (None, (None, None)),
        ("Not specified", (None, None)),
    ],
)
def test_parse_experience_range(
    raw_value: object, expected: tuple[float | None, float | None]
) -> None:
    assert parse_experience_range(raw_value) == expected


@pytest.mark.parametrize(
    ("minimum_value", "maximum_value", "expected"),
    [
        ("2 to 5yrs", None, (2.0, 5.0)),
        ("2-5 years", "", (2.0, 5.0)),
        ("2 years", "5 years", (2.0, 5.0)),
        (2, 5, (2.0, 5.0)),
        ("5+ years", None, (5.0, None)),
        (None, "up to 5 years", (None, 5.0)),
        (None, "2 to 5yrs", (2.0, 5.0)),
    ],
)
def test_parse_experience_columns(
    minimum_value: object,
    maximum_value: object,
    expected: tuple[float | None, float | None],
) -> None:
    assert parse_experience_columns(minimum_value, maximum_value) == expected


def test_dataframe_accepts_experience_range_in_minimum_column() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "Reference Number": "JOB-RANGE",
                "Job Title": "Backend Engineer",
                "Min Experience": "2 to 5yrs",
                "Max Experience": None,
            }
        ]
    )

    job = dataframe_to_jobs(dataframe)[0]

    assert job.min_experience_years == 2
    assert job.max_experience_years == 5


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
    assert job.geo == "India"
    assert job.status == "open"


def test_ats_columns_are_mapped_by_header_not_excel_position() -> None:
    dataframe = pd.DataFrame(
        [
            [
                "ignored",
                "Java; SpringBoot",
                "Backend Engineer",
                7,
                "India",
                "JOB-SHUFFLED",
                "Active",
                3,
            ]
        ],
        columns=[
            "Unrelated First Column",
            "Mandatory Skills",
            "Job Title",
            "Max Experience",
            "Geo",
            "Reference Number",
            "Job Status",
            "Min Experience",
        ],
    )

    job = dataframe_to_jobs(dataframe)[0]

    assert job.job_id == "JOB-SHUFFLED"
    assert job.title == "Backend Engineer"
    assert job.geo == "India"
    assert job.mandatory_skills == ["java", "spring boot"]
    assert job.min_experience_years == 3
    assert job.max_experience_years == 7
    assert job.status == "open"
