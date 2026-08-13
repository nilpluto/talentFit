"""Tests for skill normalization."""

import pytest

from app.skill_normalizer import (
    extract_known_skills_from_text,
    normalize_skill,
    normalize_skills,
    skills_match,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("SpringBoot", "spring boot"),
        ("PowerBi", "power bi"),
        ("Amazon Web Services", "aws"),
        ("K8s", "kubernetes"),
        ("JavaScript", "javascript"),
        ("TypeScript", "typescript"),
        ("Power BI Desktop", "power bi"),
        ("Power BI Reports", "power bi"),
        ("Service Now", "servicenow"),
        ("Dot Net Lead", ".net"),
        ("React.js", "react"),
        ("ADF", "azure data factory"),
        ("MS Fabric", "microsoft fabric"),
        ("GenAI", "generative ai"),
        ("playwrite", "playwright"),
        ("SRE", "sre"),
        ("Site Reliability Engineer", "sre"),
        ("Site Reliability Engineering", "sre"),
        ("Dev Ops", "devops"),
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


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("PowerBi", "Power BI Reports"),
        ("Service Now", "ServiceNow"),
        ("REST APIs", "RESTful API"),
        ("React JS", "React.js"),
        ("Amazon Redshift", "Redshift"),
    ],
)
def test_same_technology_variants_match(left: str, right: str) -> None:
    assert skills_match(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Java", "JavaScript"),
        ("Power BI", "Power Query"),
        ("GitHub Copilot", "Microsoft Copilot"),
        ("React", "React Native"),
    ],
)
def test_related_but_distinct_technologies_do_not_match(left: str, right: str) -> None:
    assert not skills_match(left, right)


def test_extract_known_skills_from_unstructured_resume_text() -> None:
    text = """
    Senior Site Reliability Engineer
    Operated Kubernetes services on Amazon Web Services and created Power BI reports.
    """

    assert extract_known_skills_from_text(text) == [
        "aws",
        "kubernetes",
        "power bi",
        "sre",
    ]


def test_text_extraction_does_not_match_alias_inside_an_unrelated_word() -> None:
    assert extract_known_skills_from_text("Maintained internal services") == []
