"""Extract validated candidate profiles from resume text using Ollama."""

import json
import re
from datetime import date
from typing import Any, Protocol

from ollama import Client
from pydantic import BaseModel, Field, ValidationError

from app.config import OLLAMA_HOST, OLLAMA_MODEL
from app.models import CandidateProfile
from app.skill_normalizer import normalize_skills


class ChatClient(Protocol):
    """Minimal Ollama chat interface used by candidate extraction."""

    def chat(self, **kwargs: Any) -> object: ...


class EmploymentPeriod(BaseModel):
    """One professional employment period extracted for date arithmetic."""

    start_date: str
    end_date: str | None = None


class CandidateExtraction(CandidateProfile):
    """LLM extraction payload including evidence used to verify experience."""

    employment_periods: list[EmploymentPeriod] = Field(default_factory=list)


_SYSTEM_PROMPT = """You extract factual candidate information from resume text.
Return only JSON matching the supplied schema. Examine the entire resume and infer its
meaning without expecting a fixed CV template, section order, heading vocabulary, or
layout. Information may appear in prose, lists, tables, a profile header, project
descriptions, or employment history. Do not invent information that is absent.

Field rules:
- name: the candidate's full name.
- experience_years: total professional experience as a number. Prefer an explicitly
  stated total. Otherwise calculate it from non-overlapping employment date ranges,
  ignoring education and overlapping concurrent roles. Use 0 only if unknown.
- skills: explicitly stated technical skills, tools, platforms, and frameworks.
- roles: explicitly stated job titles or professional roles, including titles in a
  resume header, employment entry, project description, or narrative sentence.
- location: an explicitly stated city, region, or country, including header text;
  otherwise null.
- summary: a concise factual professional summary based only on the resume.
- employment_periods: professional work periods only, each with start_date and
  end_date as MM/YYYY where possible. Use null for a current role. Never include
  education, training, internships presented only under education, or project dates.

Treat headings such as profile, overview, expertise, competencies, technologies,
employment, career history, assignments, and projects as possible semantic equivalents,
but rely on content rather than headings. Use empty lists for missing skills or roles
and an empty string for a missing summary."""

_EXPLICIT_EXPERIENCE_PATTERNS = [
    re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+(?:professional\s+)?experience\b"),
    re.compile(r"(?i)\btotal\s+(?:professional\s+)?experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?\b"),
    re.compile(r"(?i)\b(?:over|more than)\s+(\d+(?:\.\d+)?)\s+years?\s+(?:of\s+)?experience\b"),
]
_MONTH_YEAR = re.compile(r"(?i)^\s*(0?[1-9]|1[0-2])[/\-](\d{4})\s*$")
_YEAR = re.compile(r"^\s*(\d{4})\s*$")


def extract_explicit_experience_years(resume_text: str) -> float | None:
    """Return a prominently stated numeric total experience, when available."""
    for pattern in _EXPLICIT_EXPERIENCE_PATTERNS:
        if match := pattern.search(resume_text):
            return float(match.group(1))
    return None


def _month_index(value: str | None, *, current: date, is_end: bool) -> int | None:
    if value is None or value.strip().casefold() in {"present", "current", "now"}:
        return current.year * 12 + current.month - 1 if is_end else None
    if match := _MONTH_YEAR.match(value):
        month, year = map(int, match.groups())
        return year * 12 + month - 1
    if match := _YEAR.match(value):
        year = int(match.group(1))
        month = 12 if is_end else 1
        return year * 12 + month - 1
    return None


def calculate_employment_years(
    periods: list[EmploymentPeriod], *, current: date | None = None
) -> float | None:
    """Calculate unique professional experience without double-counting overlaps."""
    today = current or date.today()
    covered_months: set[int] = set()
    for period in periods:
        start = _month_index(period.start_date, current=today, is_end=False)
        end = _month_index(period.end_date, current=today, is_end=True)
        if start is None or end is None or end < start:
            continue
        covered_months.update(range(start, end + 1))
    return round(len(covered_months) / 12, 1) if covered_months else None


def resolve_experience_years(
    resume_text: str,
    llm_years: float,
    periods: list[EmploymentPeriod],
    *,
    current: date | None = None,
) -> float:
    """Reconcile LLM output against explicit claims and employment-date evidence."""
    explicit = extract_explicit_experience_years(resume_text)
    calculated = calculate_employment_years(periods, current=current)
    if explicit is not None:
        if calculated is not None and abs(calculated - explicit) <= 2:
            return calculated
        return explicit
    return calculated if calculated is not None else llm_years


def _response_content(response: object) -> str:
    if hasattr(response, "message"):
        message = response.message
        content = message.content if hasattr(message, "content") else message["content"]
    elif isinstance(response, dict):
        content = response.get("message", {}).get("content")
    else:
        content = None

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned no candidate profile content")
    return content.strip()


def _normalize_roles(roles: list[str]) -> list[str]:
    normalized_roles: list[str] = []
    seen: set[str] = set()
    for role in roles:
        normalized = role.strip().casefold()
        if normalized and normalized not in seen:
            normalized_roles.append(normalized)
            seen.add(normalized)
    return normalized_roles


def extract_candidate_profile(
    resume_text: str,
    *,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    client: ChatClient | None = None,
) -> CandidateProfile:
    """Use Ollama to turn resume text into a validated CandidateProfile."""
    if not resume_text.strip():
        raise ValueError("Resume text must not be empty")

    ollama_client = client or Client(host=host)
    response = ollama_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Resume text:\n\n{resume_text}"},
        ],
        format=CandidateExtraction.model_json_schema(),
        think=False,
        keep_alive="30m",
        options={"temperature": 0},
    )

    try:
        data = json.loads(_response_content(response))
    except json.JSONDecodeError as exc:
        raise ValueError("Ollama returned invalid candidate JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("Ollama candidate response must be a JSON object")

    data["skills"] = normalize_skills(data.get("skills"))
    data["roles"] = _normalize_roles(data.get("roles", []))

    try:
        extraction = CandidateExtraction.model_validate(data)
    except ValidationError as exc:
        raise ValueError("Ollama returned an invalid candidate profile") from exc

    resolved_experience = resolve_experience_years(
        resume_text,
        extraction.experience_years,
        extraction.employment_periods,
    )
    return CandidateProfile.model_validate(
        {
            **extraction.model_dump(exclude={"employment_periods"}),
            "experience_years": resolved_experience,
        }
    )
