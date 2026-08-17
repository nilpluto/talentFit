"""Tests for structured Ollama candidate extraction."""

import json
from datetime import date

import pytest

from app.candidate_extractor import (
    EmploymentPeriod,
    calculate_employment_years,
    extract_explicit_experience_years,
    extract_candidate_profile,
    resolve_experience_years,
)


class FakeChatClient:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"message": {"content": self.content}}


def test_extract_and_normalize_candidate_profile() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "name": "John Doe",
                "experience_years": 5,
                "skills": ["Java", "SpringBoot", "Amazon Web Services", "AWS"],
                "roles": ["Backend Engineer", "backend engineer"],
                "location": "Bengaluru",
                "summary": "Backend engineer building Java services.",
            }
        )
    )

    candidate = extract_candidate_profile(
        "John Doe is a backend engineer with five years of experience.",
        model="test-model",
        client=client,
    )

    assert candidate.name == "John Doe"
    assert candidate.experience_years == 5
    assert candidate.skills == ["java", "spring boot", "aws"]
    assert candidate.roles == ["backend engineer"]
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["options"] == {"temperature": 0}
    assert client.calls[0]["think"] is False
    assert client.calls[0]["keep_alive"] == "30m"
    assert isinstance(client.calls[0]["format"], dict)
    system_prompt = client.calls[0]["messages"][0]["content"]
    assert "without expecting a fixed CV template" in system_prompt
    assert "non-overlapping employment date ranges" in system_prompt


def test_reject_empty_resume_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        extract_candidate_profile("  ")


def test_reject_invalid_json_response() -> None:
    with pytest.raises(ValueError, match="invalid candidate JSON"):
        extract_candidate_profile("Resume", client=FakeChatClient("not JSON"))


def test_reject_non_object_json_response() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        extract_candidate_profile("Resume", client=FakeChatClient("[]"))


def test_reject_profile_that_fails_validation() -> None:
    response = json.dumps({"name": "", "experience_years": -1})

    with pytest.raises(ValueError, match="invalid candidate profile"):
        extract_candidate_profile("Resume", client=FakeChatClient(response))


def test_reject_response_without_content() -> None:
    with pytest.raises(RuntimeError, match="no candidate profile content"):
        extract_candidate_profile("Resume", client=FakeChatClient(None))


def test_calculate_employment_experience_merges_overlapping_periods() -> None:
    periods = [
        EmploymentPeriod(start_date="10/2014", end_date="05/2017"),
        EmploymentPeriod(start_date="05/2017", end_date="02/2021"),
        EmploymentPeriod(start_date="03/2021", end_date="12/2025"),
        EmploymentPeriod(start_date="12/2025", end_date=None),
    ]

    assert calculate_employment_years(
        periods, current=date(2026, 8, 11)
    ) == 11.9


def test_explicit_total_blocks_education_dates_from_inflating_experience() -> None:
    resume_text = "Java Tech Lead with 11+ years of experience. Education 06/2010 - 05/2014."
    periods_with_education_mistake = [
        EmploymentPeriod(start_date="06/2010", end_date=None)
    ]

    assert resolve_experience_years(
        resume_text,
        llm_years=15,
        periods=periods_with_education_mistake,
        current=date(2026, 8, 11),
    ) == 11


def test_written_explicit_experience_is_recognized() -> None:
    resume_text = "With more than ten years of experience, I design distributed systems."

    assert extract_explicit_experience_years(resume_text) == 10
    assert resolve_experience_years(
        resume_text,
        llm_years=14,
        periods=[EmploymentPeriod(start_date="01/2014", end_date="07/2018")],
        current=date(2026, 8, 13),
    ) == 10


def test_incomplete_periods_do_not_replace_consistent_llm_total() -> None:
    periods = [EmploymentPeriod(start_date="01/2014", end_date="07/2018")]

    assert resolve_experience_years(
        "Experienced software architect.",
        llm_years=14,
        periods=periods,
        current=date(2026, 8, 13),
    ) == 14


def test_complete_periods_verify_close_llm_total() -> None:
    periods = [EmploymentPeriod(start_date="01/2020", end_date=None)]

    assert resolve_experience_years(
        "Experienced engineer.",
        llm_years=6,
        periods=periods,
        current=date(2026, 8, 13),
    ) == 6.7


def test_candidate_extraction_uses_verified_employment_periods() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "name": "Nilanjan Jha",
                "experience_years": 15,
                "skills": ["Java"],
                "roles": ["Tech Lead"],
                "summary": "Java Tech Lead with 11+ years of experience.",
                "employment_periods": [
                    {"start_date": "10/2014", "end_date": "05/2017"},
                    {"start_date": "05/2017", "end_date": "02/2021"},
                    {"start_date": "03/2021", "end_date": "12/2025"},
                    {"start_date": "12/2025", "end_date": None},
                ],
            }
        )
    )

    candidate = extract_candidate_profile(
        "Java Tech Lead with 11+ years of experience.", client=client
    )

    assert 11 <= candidate.experience_years <= 12


def test_candidate_extraction_recovers_skills_from_full_resume_evidence() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "name": "Platform Candidate",
                "experience_years": 8,
                "skills": ["DevOps"],
                "roles": [],
                "summary": "Platform engineer.",
            }
        )
    )

    candidate = extract_candidate_profile(
        "Senior Site Reliability Engineer operating Kubernetes on AWS.",
        client=client,
    )

    assert candidate.skills == ["devops", "aws", "kubernetes", "sre"]
