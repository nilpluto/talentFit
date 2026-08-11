"""Tests for candidate-to-job semantic retrieval."""

from uuid import uuid4

import chromadb

from app.models import CandidateProfile, Job
from app.search_service import search_candidate_jobs
from app.vector_store import JobVectorStore


def test_candidate_retrieval_returns_nearest_jobs() -> None:
    store = JobVectorStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"talentfit-search-test-{uuid4().hex}",
    )
    store.upsert_jobs(
        [
            Job(job_id="JAVA", title="Java Backend Engineer"),
            Job(job_id="PYTHON", title="Python Django Developer"),
        ],
        ["Java backend engineer", "Python Django developer"],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    candidate = CandidateProfile(
        name="Candidate",
        experience_years=5,
        skills=["java", "spring boot"],
        roles=["backend engineer"],
    )

    hits = search_candidate_jobs(
        candidate,
        vector_store=store,
        embedder=lambda document: [1.0, 0.0],
        limit=2,
    )

    assert [hit.job.job_id for hit in hits] == ["JAVA", "PYTHON"]
    assert hits[0].semantic_score > hits[1].semantic_score


def test_candidate_retrieval_filters_open_and_referral_jobs() -> None:
    store = JobVectorStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"talentfit-filter-test-{uuid4().hex}",
    )
    jobs = [
        Job(job_id="OPEN-REF", title="Open referral", status="open", referral_allowed=True),
        Job(job_id="OPEN-NO", title="Open no referral", status="open", referral_allowed=False),
        Job(job_id="CLOSED-REF", title="Closed referral", status="closed", referral_allowed=True),
    ]
    store.upsert_jobs(
        jobs,
        [job.title for job in jobs],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )
    candidate = CandidateProfile(name="Candidate", experience_years=5)

    hits = search_candidate_jobs(
        candidate,
        vector_store=store,
        embedder=lambda document: [1.0, 0.0],
        limit=10,
        open_only=True,
        referral_only=True,
    )

    assert [hit.job.job_id for hit in hits] == ["OPEN-REF"]
