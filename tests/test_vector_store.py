"""Tests for the isolated ChromaDB job vector store."""

from uuid import uuid4

import chromadb
import pytest

from app.models import Job
from app.vector_store import JobVectorStore


@pytest.fixture
def vector_store() -> JobVectorStore:
    return JobVectorStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"talentfit-test-{uuid4().hex}",
    )


def test_new_store_is_empty(vector_store: JobVectorStore) -> None:
    assert vector_store.count_jobs() == 0
    assert vector_store.search_jobs([1.0, 0.0]) == []


def test_clear_jobs_removes_all_records(vector_store: JobVectorStore) -> None:
    vector_store.upsert_jobs(
        [
            Job(job_id="JOB-001", title="First"),
            Job(job_id="JOB-002", title="Second"),
        ],
        ["First document", "Second document"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert vector_store.clear_jobs() == 2
    assert vector_store.count_jobs() == 0
    assert vector_store.clear_jobs() == 0


def test_list_and_delete_selected_jobs(vector_store: JobVectorStore) -> None:
    vector_store.upsert_jobs(
        [Job(job_id="JOB-001", title="First"), Job(job_id="JOB-002", title="Second")],
        ["First document", "Second document"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert set(vector_store.list_job_ids()) == {"JOB-001", "JOB-002"}
    assert vector_store.delete_jobs(["JOB-001", "MISSING"]) == 1
    assert vector_store.list_job_ids() == ["JOB-002"]


def test_upsert_and_count_jobs(vector_store: JobVectorStore) -> None:
    job = Job(job_id="JOB-001", title="Java Engineer")

    vector_store.upsert_job(job, "Java backend APIs", [1.0, 0.0])

    assert vector_store.count_jobs() == 1


def test_upsert_replaces_job_with_same_id(vector_store: JobVectorStore) -> None:
    vector_store.upsert_job(
        Job(job_id="JOB-001", title="Old Title"), "Old document", [1.0, 0.0]
    )
    vector_store.upsert_job(
        Job(job_id="JOB-001", title="New Title"), "New document", [1.0, 0.0]
    )

    hits = vector_store.search_jobs([1.0, 0.0])

    assert vector_store.count_jobs() == 1
    assert hits[0].job.title == "New Title"
    assert hits[0].document == "New document"


def test_search_returns_nearest_job_first(vector_store: JobVectorStore) -> None:
    jobs = [
        Job(job_id="JAVA", title="Java Backend Engineer"),
        Job(job_id="PYTHON", title="Python Django Developer"),
    ]
    vector_store.upsert_jobs(
        jobs,
        ["Java backend engineer", "Python Django developer"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    hits = vector_store.search_jobs([0.9, 0.1], limit=2)

    assert [hit.job.job_id for hit in hits] == ["JAVA", "PYTHON"]
    assert hits[0].semantic_score > hits[1].semantic_score


def test_batch_lengths_must_match(vector_store: JobVectorStore) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        vector_store.upsert_jobs(
            [Job(job_id="JOB-001", title="Engineer")],
            [],
            [[1.0, 0.0]],
        )


def test_content_hashes_are_available_after_upsert(vector_store: JobVectorStore) -> None:
    job = Job(job_id="JOB-001", title="Engineer")

    vector_store.upsert_job(
        job,
        "Engineer document",
        [1.0, 0.0],
        content_hash="known-hash",
    )

    assert vector_store.get_content_hashes(["JOB-001", "MISSING"]) == {
        "JOB-001": "known-hash"
    }


def test_legacy_records_gain_status_filter_metadata() -> None:
    client = chromadb.EphemeralClient()
    collection_name = f"talentfit-legacy-test-{uuid4().hex}"
    collection = client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    legacy_job = Job(
        job_id="LEGACY",
        title="Legacy open job",
        status="open",
    )
    collection.upsert(
        ids=[legacy_job.job_id],
        documents=[legacy_job.title],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"job_json": legacy_job.model_dump_json(), "content_hash": "old"}],
    )

    migrated_store = JobVectorStore(client=client, collection_name=collection_name)
    hits = migrated_store.search_jobs([1.0, 0.0], open_only=True)

    assert [hit.job.job_id for hit in hits] == ["LEGACY"]


def test_get_jobs_applies_open_filter(vector_store: JobVectorStore) -> None:
    jobs = [
        Job(job_id="OPEN-ONE", title="Open one", status="open"),
        Job(job_id="OPEN-TWO", title="Open two", status="open"),
        Job(job_id="CLOSED", title="Closed", status="closed"),
    ]
    vector_store.upsert_jobs(
        jobs,
        [job.title for job in jobs],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )

    assert [job.job_id for job in vector_store.get_jobs(open_only=True)] == [
        "OPEN-ONE",
        "OPEN-TWO",
    ]


def test_india_filter_excludes_other_and_missing_geographies(
    vector_store: JobVectorStore,
) -> None:
    jobs = [
        Job(job_id="INDIA", title="India job", geo="India", status="open"),
        Job(job_id="INDIA-UPPER", title="India uppercase", geo="INDIA", status="open"),
        Job(job_id="USA", title="USA job", geo="USA", status="open"),
        Job(job_id="MISSING", title="Missing Geo", status="open"),
    ]
    vector_store.upsert_jobs(
        jobs,
        [job.title for job in jobs],
        [[1.0, 0.0], [0.9, 0.1], [0.99, 0.01], [0.98, 0.02]],
    )

    assert [job.job_id for job in vector_store.get_jobs(india_only=True)] == [
        "INDIA",
        "INDIA-UPPER",
    ]
    assert [
        hit.job.job_id
        for hit in vector_store.search_jobs(
            [1.0, 0.0], limit=10, open_only=True, india_only=True
        )
    ] == ["INDIA", "INDIA-UPPER"]


def test_reject_invalid_search_arguments(vector_store: JobVectorStore) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        vector_store.search_jobs([])

    with pytest.raises(ValueError, match="greater than zero"):
        vector_store.search_jobs([1.0, 0.0], limit=0)
