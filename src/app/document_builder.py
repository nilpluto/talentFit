"""Build stable text documents from TalentFit models for semantic search."""

from collections.abc import Iterable

from app.models import CandidateProfile, Job


def _format_years(years: float) -> str:
    return f"{years:g}"


def format_experience(job: Job) -> str:
    """Format a job's experience range as readable search text."""
    minimum = job.min_experience_years
    maximum = job.max_experience_years

    if minimum is None and maximum is None:
        return "Not specified"
    if minimum is None:
        return f"Up to {_format_years(maximum)} years"
    if maximum is None:
        return f"{_format_years(minimum)}+ years"
    if minimum == maximum:
        return f"{_format_years(minimum)} years"
    return f"{_format_years(minimum)} to {_format_years(maximum)} years"


def _format_skills(skills: list[str]) -> str:
    return ", ".join(skills) if skills else "None specified"


def build_job_document(job: Job) -> str:
    """Build embedding text exclusively from the supported ATS fields."""
    return "\n".join(
        [
            f"Reference Number: {job.job_id}",
            f"Job Title: {job.title}",
            f"Designation: {job.designation or 'Not specified'}",
            f"Geo: {job.geo or 'Not specified'}",
            f"Business Unit: {job.business_unit or 'Not specified'}",
            f"Mandatory Skills: {_format_skills(job.mandatory_skills)}",
            f"Experience: {format_experience(job)}",
            f"Job Status: {job.status}",
        ]
    )


def build_job_documents(jobs: Iterable[Job]) -> list[str]:
    """Build embedding documents for a collection of jobs in input order."""
    return [build_job_document(job) for job in jobs]


def build_candidate_document(candidate: CandidateProfile) -> str:
    """Convert a candidate profile into consistent semantic-search text."""
    return "\n".join(
        [
            f"Candidate Roles: {_format_skills(candidate.roles)}",
            f"Candidate Skills: {_format_skills(candidate.skills)}",
            f"Experience: {_format_years(candidate.experience_years)} years",
            f"Location: {candidate.location or 'Not specified'}",
            f"Summary: {candidate.summary or 'Not specified'}",
        ]
    )
