"""Utilities for converting skill names to consistent canonical values."""

import re
from collections.abc import Iterable


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[_-]+")
_WHITESPACE = re.compile(r"\s+")

_SKILL_ALIASES = {
    ".net": ".net",
    ".net developer": ".net",
    "dot net": ".net",
    "dot net lead": ".net",
    "adf": "azure data factory",
    "ai": "artificial intelligence",
    "amazon web service": "aws",
    "amazon web services": "aws",
    "amazon aws": "aws",
    "amazon redshift": "redshift",
    "artificial intelligence": "artificial intelligence",
    "aws": "aws",
    "azure data factory": "azure data factory",
    "ci/cd": "ci/cd",
    "ci cd": "ci/cd",
    "continuous integration and continuous delivery": "ci/cd",
    "gcp": "gcp",
    "gen ai": "generative ai",
    "genai": "generative ai",
    "generative ai": "generative ai",
    "google cloud platform": "gcp",
    "graphana": "grafana",
    "java script": "javascript",
    "java development": "java",
    "java developer": "java",
    "javascript": "javascript",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "lightning web components": "lightning web components",
    "lightning web components (lwc)": "lightning web components",
    "lwc": "lightning web components",
    "microsoft azure": "azure",
    "microsoft fabric": "microsoft fabric",
    "ms fabric": "microsoft fabric",
    "node js": "node.js",
    "node.js": "node.js",
    "nodejs": "node.js",
    "playwright automation": "playwright",
    "playwrite": "playwright",
    "powerbi": "power bi",
    "power bi": "power bi",
    "power bi desktop": "power bi",
    "power bi reports": "power bi",
    "python development": "python",
    "python developer": "python",
    "python software developer": "python",
    "react js": "react",
    "react js developer": "react",
    "react.js": "react",
    "reactjs": "react",
    "redshift": "redshift",
    "rest api": "rest api",
    "rest api development": "rest api",
    "rest apis": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "service now": "servicenow",
    "servicenow": "servicenow",
    "servicenow architect": "servicenow",
    "servicenow developer": "servicenow",
    "servicenow development": "servicenow",
    "springboot": "spring boot",
    "spring boot": "spring boot",
    "type script": "typescript",
    "typescript": "typescript",
}


def _compact_skill(skill: str) -> str:
    """Remove presentation punctuation when comparing canonical skill names."""
    return re.sub(r"[^a-z0-9+#]+", "", skill)


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


def skills_match(left: str | None, right: str | None) -> bool:
    """Return whether two labels are safe variants of the same technology."""
    normalized_left = normalize_skill(left)
    normalized_right = normalize_skill(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    return _compact_skill(normalized_left) == _compact_skill(normalized_right)
