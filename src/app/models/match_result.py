"""Match result model."""

from pydantic import Field

from app.models.base import TalentFitModel
from app.models.job import Job


class MatchResult(TalentFitModel):
    """An explainable match between a candidate and a job."""

    job: Job
    semantic_score: float = Field(ge=0, le=1)
    mandatory_score: float = Field(default=0, ge=0, le=100)
    optional_score: float = Field(default=0, ge=0, le=100)
    experience_score: float = Field(default=0, ge=0, le=100)
    final_score: float = Field(ge=0, le=100)
    matched_mandatory: list[str] = Field(default_factory=list)
    missing_mandatory: list[str] = Field(default_factory=list)
    matched_optional: list[str] = Field(default_factory=list)
    missing_optional: list[str] = Field(default_factory=list)
