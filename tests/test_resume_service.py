"""Tests for PDF resume text extraction."""

from pathlib import Path

import pymupdf
import pytest

from app.resume_service import extract_resume_text


SAMPLE_RESUME = Path(__file__).parents[1] / "resources" / "sample_resume.pdf"


def _create_pdf(path: Path, page_texts: list[str]) -> None:
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)


def test_extract_text_from_multi_page_resume(tmp_path: Path) -> None:
    resume_path = tmp_path / "resume.pdf"
    _create_pdf(
        resume_path,
        [
            "John Doe\nBackend Engineer",
            "Skills: Java, Spring Boot, AWS\nExperience: 5 years",
        ],
    )

    text = extract_resume_text(resume_path)

    assert "John Doe" in text
    assert "Backend Engineer" in text
    assert "Skills: Java, Spring Boot, AWS" in text
    assert "Experience: 5 years" in text


def test_permanent_sample_resume_is_extractable() -> None:
    text = extract_resume_text(SAMPLE_RESUME)

    assert "John Doe" in text
    assert "Backend Engineer" in text
    assert "Java, Spring Boot" in text
    assert "5 years" in text


@pytest.mark.parametrize(
    ("filename", "expected_name", "expected_skill"),
    [
        ("sample_resume_data_ai_engineer.pdf", "AARAV MEHTA", "Snowflake"),
        ("sample_resume_cloud_sre.pdf", "MAYA RAO", "Kubernetes"),
    ],
)
def test_additional_sample_resumes_are_extractable(
    filename: str, expected_name: str, expected_skill: str
) -> None:
    path = SAMPLE_RESUME.parent / filename

    text = extract_resume_text(path)

    assert expected_name in text
    assert expected_skill in text
    assert "Present" in text


def test_reject_non_pdf_file(tmp_path: Path) -> None:
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("resume", encoding="utf-8")

    with pytest.raises(ValueError, match="PDF"):
        extract_resume_text(resume_path)


def test_reject_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        extract_resume_text(tmp_path / "missing.pdf")


def test_reject_pdf_without_extractable_text(tmp_path: Path) -> None:
    resume_path = tmp_path / "scanned.pdf"
    _create_pdf(resume_path, [""])

    with pytest.raises(ValueError, match="require OCR"):
        extract_resume_text(resume_path)


def test_reject_corrupted_pdf(tmp_path: Path) -> None:
    resume_path = tmp_path / "corrupted.pdf"
    resume_path.write_bytes(b"not a real PDF")

    with pytest.raises(ValueError, match="Invalid or corrupted"):
        extract_resume_text(resume_path)
