"""Read ATS exports and convert their rows into TalentFit job models."""

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from app.models import Job
from app.skill_normalizer import normalize_skills


SUPPORTED_FILE_TYPES = {".csv", ".xls", ".xlsx"}
REQUIRED_COLUMNS = {"reference_number", "job_title"}
ATS_DISPLAY_COLUMNS = [
    "reference_number",
    "job_created_date",
    "job_title",
    "open_positions",
    "designation",
    "geo",
    "business_unit",
    "min_experience",
    "max_experience",
    "mandatory_skills",
    "job_status",
    "referral_enabled",
]
ATS_DISPLAY_LABELS = {
    "reference_number": "Reference Number",
    "job_created_date": "Job Created Date",
    "job_title": "Job Title",
    "open_positions": "Open Positions",
    "designation": "Designation",
    "geo": "Geo",
    "business_unit": "Business Unit",
    "min_experience": "Min Experience",
    "max_experience": "Max Experience",
    "mandatory_skills": "Mandatory Skills",
    "job_status": "Job Status",
    "referral_enabled": "Referral Enabled",
}
_COLUMN_ALIASES = {
    "job_id": "reference_number",
    "reference_number": "reference_number",
    "job_code": "reference_number",
    "title": "job_title",
    "job_title": "job_title",
    "created_date": "job_created_date",
    "job_created_date": "job_created_date",
    "open_position": "open_positions",
    "open_positions": "open_positions",
    "designation": "designation",
    "geo": "geo",
    "business_unit": "business_unit",
    "min_exp": "min_experience",
    "min_experience": "min_experience",
    "min_experience_years": "min_experience",
    "max_exp": "max_experience",
    "max_experience": "max_experience",
    "max_experience_years": "max_experience",
    "mandatory_skills": "mandatory_skills",
    "job_status": "job_status",
    "status": "job_status",
    "referral_enabled": "referral_enabled",
    "refferal_enabled": "referral_enabled",
    "referral_allowed": "referral_enabled",
}
_SKILL_SEPARATOR = re.compile(r"[;,|]")
_EXPERIENCE_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")
_TRUE_VALUES = {"1", "true", "yes", "y"}
_FALSE_VALUES = {"0", "false", "no", "n", ""}
_STATUS_ALIASES = {
    "active": "open",
    "open": "open",
    "published": "open",
    "closed": "closed",
    "filled": "closed",
    "inactive": "closed",
    "archived": "closed",
    "draft": "draft",
    "hold": "on hold",
    "on hold": "on hold",
    "paused": "on hold",
}


def read_ats_file(file_path: str | Path) -> pd.DataFrame:
    """Read a supported ATS export into a DataFrame."""
    path = Path(file_path)
    suffix = path.suffix.casefold()

    if suffix not in SUPPORTED_FILE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_FILE_TYPES))
        raise ValueError(f"Unsupported ATS file type {suffix!r}; expected one of: {supported}")

    if not path.is_file():
        raise FileNotFoundError(f"ATS file not found: {path}")

    if suffix == ".csv":
        dataframe = pd.read_csv(path)
    else:
        dataframe = pd.read_excel(path)

    dataframe = normalize_ats_columns(dataframe)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ATS file is missing required columns: {missing}")

    return dataframe


def _column_key(value: object) -> str:
    """Convert a human-readable ATS header into a stable lookup key."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def normalize_ats_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Map supported header variants and discard unrelated ATS columns."""
    selected: dict[str, pd.Series] = {}
    for position, source_name in enumerate(dataframe.columns):
        canonical = _COLUMN_ALIASES.get(_column_key(source_name))
        if canonical is not None and canonical not in selected:
            selected[canonical] = dataframe.iloc[:, position]

    missing_columns = REQUIRED_COLUMNS.difference(selected)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ATS data is missing required columns: {missing}")

    normalized = pd.DataFrame(selected, index=dataframe.index)
    for column in ATS_DISPLAY_COLUMNS:
        if column not in normalized:
            normalized[column] = None
    return normalized[ATS_DISPLAY_COLUMNS]


def ats_preview(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return recruiter-facing ATS columns with readable labels."""
    return normalize_ats_columns(dataframe).rename(columns=ATS_DISPLAY_LABELS)


def _text(value: object, default: str = "") -> str:
    """Convert a DataFrame cell to stripped text without leaking NaN."""
    if pd.isna(value):
        return default
    return str(value).strip()


def _optional_text(value: object) -> str | None:
    text = _text(value)
    return text or None


def _optional_date_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime("%d-%b-%Y")
    return _optional_text(value)


def _optional_float(value: object) -> float | None:
    if pd.isna(value) or value == "":
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def clean_description(value: object) -> str:
    """Convert an HTML or plain-text job description into clean text."""
    text = _text(value)
    if not text:
        return ""

    plain_text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return _WHITESPACE.sub(" ", plain_text).strip()


def parse_experience_range(value: object) -> tuple[float | None, float | None]:
    """Parse common ATS experience formats into minimum and maximum years."""
    if pd.isna(value) or value == "":
        return None, None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        years = float(value)
        return years, years

    text = str(value).strip().casefold()
    numbers = [float(number) for number in _EXPERIENCE_NUMBER.findall(text)]
    if not numbers:
        return None, None

    minimum = numbers[0]
    if len(numbers) >= 2:
        return minimum, numbers[1]
    if "+" in text or "minimum" in text or "min" in text:
        return minimum, None
    return minimum, minimum


def _skill_list(value: object) -> list[str]:
    text = _text(value)
    if not text:
        return []
    return normalize_skills(_SKILL_SEPARATOR.split(text))


def normalize_referral_flag(value: object) -> bool:
    """Normalize common ATS boolean representations for referral eligibility."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return False


def normalize_job_status(value: object) -> str:
    """Map common ATS status labels to stable canonical values."""
    normalized = _text(value, default="unknown").casefold()
    return _STATUS_ALIASES.get(normalized, normalized or "unknown")


def dataframe_to_jobs(dataframe: pd.DataFrame) -> list[Job]:
    """Convert ATS DataFrame rows into validated Job objects."""
    dataframe = normalize_ats_columns(dataframe)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ATS data is missing required columns: {missing}")

    jobs: list[Job] = []
    for row in dataframe.to_dict(orient="records"):
        minimum_experience = _optional_float(row.get("min_experience"))
        maximum_experience = _optional_float(row.get("max_experience"))

        jobs.append(
            Job(
                job_id=_text(row.get("reference_number")),
                title=_text(row.get("job_title")),
                created_date=_optional_date_text(row.get("job_created_date")),
                open_positions=_optional_int(row.get("open_positions")),
                designation=_optional_text(row.get("designation")),
                geo=_optional_text(row.get("geo")),
                business_unit=_optional_text(row.get("business_unit")),
                mandatory_skills=_skill_list(row.get("mandatory_skills")),
                min_experience_years=minimum_experience,
                max_experience_years=maximum_experience,
                status=normalize_job_status(row.get("job_status")),
                referral_allowed=normalize_referral_flag(row.get("referral_enabled")),
            )
        )

    return jobs


def load_jobs(file_path: str | Path) -> list[Job]:
    """Load a CSV or Excel ATS export as validated Job objects."""
    return dataframe_to_jobs(read_ats_file(file_path))
