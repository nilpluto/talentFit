"""Extract plain text from candidate resume PDFs."""

import re
from pathlib import Path

import pymupdf


_EXTRA_SPACES = re.compile(r"[ \t]+")
_EXTRA_NEWLINES = re.compile(r"\n{3,}")


def _clean_page_text(text: str) -> str:
    lines = [_EXTRA_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    return _EXTRA_NEWLINES.sub("\n\n", cleaned)


def extract_resume_text(file_path: str | Path) -> str:
    """Extract readable text from every page of a PDF resume."""
    path = Path(file_path)
    if path.suffix.casefold() != ".pdf":
        raise ValueError("Resume must be a PDF file")
    if not path.is_file():
        raise FileNotFoundError(f"Resume PDF not found: {path}")

    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError("Password-protected resume PDFs are not supported")

            pages = [
                page_text
                for page in document
                if (page_text := _clean_page_text(page.get_text("text", sort=True)))
            ]
    except pymupdf.FileDataError as exc:
        raise ValueError(f"Invalid or corrupted PDF: {path}") from exc

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(
            "Resume PDF contains no extractable text; scanned PDFs require OCR"
        )
    return text
