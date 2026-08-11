"""Utilities for converting skill names to consistent canonical values."""

import re
from collections.abc import Iterable


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[_-]+")
_WHITESPACE = re.compile(r"\s+")

_SKILL_ALIASES = {
    "amazon web service": "aws",
    "amazon web services": "aws",
    "amazon aws": "aws",
    "aws": "aws",
    "java script": "javascript",
    "javascript": "javascript",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "powerbi": "power bi",
    "power bi": "power bi",
    "springboot": "spring boot",
    "spring boot": "spring boot",
    "type script": "typescript",
    "typescript": "typescript",
}


def normalize_skill(skill: str | None) -> str:
    """Return the canonical lowercase representation of one skill."""
    if skill is None:
        return ""

    normalized = _CAMEL_CASE_BOUNDARY.sub(" ", str(skill).strip())
    normalized = _SEPARATORS.sub(" ", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).casefold().strip()

    if not normalized:
        return ""

    return _SKILL_ALIASES.get(normalized, normalized)


def normalize_skills(skills: Iterable[str | None] | None) -> list[str]:
    """Normalize a skill collection and remove duplicates in input order."""
    if skills is None:
        return []

    normalized_skills: list[str] = []
    seen: set[str] = set()

    for skill in skills:
        normalized = normalize_skill(skill)
        if normalized and normalized not in seen:
            normalized_skills.append(normalized)
            seen.add(normalized)

    return normalized_skills
