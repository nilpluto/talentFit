"""Job model."""

from pydantic import Field, field_validator, model_validator

from app.models.base import TalentFitModel


class Job(TalentFitModel):
    """A normalized job imported from the ATS."""

    job_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    designation: str | None = None
    geo: str = "India"
    business_unit: str | None = None
    description: str = ""
    mandatory_skills: list[str] = Field(default_factory=list)
    optional_skills: list[str] = Field(default_factory=list)
    min_experience_years: float | None = Field(default=None, ge=0)
    max_experience_years: float | None = Field(default=None, ge=0)
    location: str | None = None
    status: str = "unknown"

    @field_validator("geo", mode="before")
    @classmethod
    def default_missing_geo_to_india(cls, value: object) -> str:
        """Treat a missing or blank ATS Geo as India for the MVP."""
        if value is None or not str(value).strip():
            return "India"
        return str(value).strip()

    @property
    def is_open(self) -> bool:
        """Treat every status except the canonical closed status as open."""
        return self.status.strip().casefold() != "closed"

    @model_validator(mode="after")
    def validate_experience_range(self) -> "Job":
        """Ensure the maximum experience is not below the minimum."""
        if (
            self.min_experience_years is not None
            and self.max_experience_years is not None
            and self.max_experience_years < self.min_experience_years
        ):
            raise ValueError(
                "max_experience_years must be greater than or equal to "
                "min_experience_years"
            )
        return self
