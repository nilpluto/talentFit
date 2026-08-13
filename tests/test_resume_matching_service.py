"""Tests for the complete resume matching orchestration."""

from pathlib import Path

import pytest

import app.resume_matching_service as service
from app.models import CandidateProfile, Job, MatchResult
from app.vector_store import JobSearchHit


def test_match_resume_connects_all_pipeline_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CandidateProfile(
        name="Test Candidate",
        experience_years=5,
        skills=["java", "spring boot"],
        roles=["backend engineer"],
    )
    job = Job(
        job_id="JOB-001",
        title="Java Backend Engineer",
        mandatory_skills=["java", "spring boot"],
    )
    hit = JobSearchHit(
        job=job,
        document="Java backend engineer",
        distance=0.1,
        semantic_score=0.9,
    )
    match = MatchResult(
        job=job,
        semantic_score=0.9,
        mandatory_score=100,
        optional_score=100,
        experience_score=100,
        final_score=98,
        matched_mandatory=["java", "spring boot"],
    )
    calls: list[object] = []

    monkeypatch.setattr(
        service,
        "extract_resume_text",
        lambda path: calls.append(("pdf", path)) or "resume text",
    )
    monkeypatch.setattr(
        service,
        "extract_candidate_profile",
        lambda text: calls.append(("candidate", text)) or candidate,
    )
    monkeypatch.setattr(
        service,
        "embed_text",
        lambda document: calls.append(("embed", document)) or [1.0, 0.0],
    )

    class FakeStore:
        def search_jobs(self, embedding, *, limit, open_only, india_only):
            calls.append(("search", embedding, limit, open_only, india_only))
            return [hit]

    monkeypatch.setattr(service, "JobVectorStore", FakeStore)
    monkeypatch.setattr(
        service,
        "rank_job_matches",
        lambda profile, hits, limit: calls.append(("rank", profile, hits, limit))
        or [match],
    )

    summary = service.match_resume(
        Path("resume.pdf"),
        retrieval_limit=10,
        result_limit=3,
        open_only=True,
    )

    assert summary.candidate == candidate
    assert summary.matches == [match]
    assert calls == [
        ("pdf", Path("resume.pdf")),
        ("candidate", "resume text"),
        ("embed", service.build_candidate_document(candidate)),
        ("search", [1.0, 0.0], 10, True, True),
        ("rank", candidate, [hit], 3),
    ]


@pytest.mark.parametrize(
    ("retrieval_limit", "result_limit", "message"),
    [(0, 3, "Retrieval"), (10, 0, "Result")],
)
def test_match_resume_rejects_invalid_limits(
    retrieval_limit: int, result_limit: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        service.match_resume(
            "resume.pdf",
            retrieval_limit=retrieval_limit,
            result_limit=result_limit,
        )


class DiagnosticStore:
    def __init__(self, jobs: list[Job], hits: list[JobSearchHit]) -> None:
        self.jobs = jobs
        self.hits = hits

    def get_jobs(self) -> list[Job]:
        return self.jobs

    def search_jobs(self, embedding, *, limit, open_only, india_only):
        assert embedding == [1.0, 0.0]
        assert india_only is True
        return self.hits[:limit]


def test_explain_job_selection_reports_minimum_experience_failure() -> None:
    candidate = CandidateProfile(
        name="Candidate",
        experience_years=4.5,
        skills=["backend", "system design"],
    )
    job = Job(
        job_id="19388",
        title="Backend TA - GCC",
        mandatory_skills=["backend", "programming and design"],
        min_experience_years=9,
        max_experience_years=12,
        status="open",
    )
    hit = JobSearchHit(
        job=job,
        document="Backend architect",
        distance=0.4,
        semantic_score=0.6,
    )

    diagnostic = service.explain_job_selection(
        service.PreparedResume(candidate=candidate, candidate_embedding=[1.0, 0.0]),
        " 19388 ",
        [],
        open_only=True,
        vector_store=DiagnosticStore([job], [hit]),
    )

    assert diagnostic.outcome == "minimum_experience"
    assert diagnostic.semantic_rank == 1
    assert diagnostic.experience_passed is False
    assert diagnostic.match is not None
    assert diagnostic.match.matched_mandatory == ["backend"]
    assert "4.5 years" in diagnostic.explanation
    assert "9-year minimum" in diagnostic.explanation


def test_explain_job_selection_reports_selected_rank() -> None:
    candidate = CandidateProfile(
        name="Candidate",
        experience_years=8,
        skills=["sre"],
    )
    job = Job(
        job_id="19745",
        title="Site Reliability Engineer",
        mandatory_skills=["sre"],
        min_experience_years=7,
        status="open",
    )
    hit = JobSearchHit(
        job=job,
        document="Site reliability engineer",
        distance=0.1,
        semantic_score=0.9,
    )
    selected = service.match_candidate_to_job(candidate, job, 0.9)

    diagnostic = service.explain_job_selection(
        service.PreparedResume(candidate=candidate, candidate_embedding=[1.0, 0.0]),
        "19745",
        [selected],
        open_only=True,
        vector_store=DiagnosticStore([job], [hit]),
    )

    assert diagnostic.outcome == "selected"
    assert diagnostic.final_rank == 1
    assert diagnostic.experience_passed is True
    assert diagnostic.match is not None
    assert diagnostic.match.matched_mandatory == ["sre"]


def test_explain_job_selection_reports_missing_reference() -> None:
    prepared = service.PreparedResume(
        candidate=CandidateProfile(name="Candidate", experience_years=5),
        candidate_embedding=[1.0, 0.0],
    )

    diagnostic = service.explain_job_selection(
        prepared,
        "UNKNOWN",
        [],
        vector_store=DiagnosticStore([], []),
    )

    assert diagnostic.outcome == "not_found"
    assert diagnostic.job is None
