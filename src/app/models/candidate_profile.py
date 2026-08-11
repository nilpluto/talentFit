"""Candidate profile model."""

from pydantic import Field

from app.models.base import TalentFitModel


class CandidateProfile(TalentFitModel):
    """Structured candidate information used for search and matching."""

    name: str = Field(min_length=1)
    experience_years: float = Field(ge=0)
    skills: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    location: str | None = None
    summary: str = ""
