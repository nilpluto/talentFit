"""Tests for skill normalization."""

import pytest

from app.skill_normalizer import normalize_skill, normalize_skills


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("SpringBoot", "spring boot"),
        ("PowerBi", "power bi"),
        ("Amazon Web Services", "aws"),
        ("K8s", "kubernetes"),
        ("JavaScript", "javascript"),
        ("TypeScript", "typescript"),
    ],
)
def test_known_skill_variants(source: str, expected: str) -> None:
    assert normalize_skill(source) == expected


def test_casing_whitespace_and_separators() -> None:
    assert normalize_skill("  MACHINE_learning  ") == "machine learning"
    assert normalize_skill("spring-boot") == "spring boot"
    assert normalize_skill("REST   API") == "rest api"


def test_empty_values() -> None:
    assert normalize_skill(None) == ""
    assert normalize_skill("   ") == ""
    assert normalize_skills(None) == []


def test_list_normalization_removes_duplicates_in_order() -> None:
    skills = ["AWS", "Amazon Web Services", "K8s", "kubernetes", None, "Java"]

    assert normalize_skills(skills) == ["aws", "kubernetes", "java"]
