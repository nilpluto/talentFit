"""Tests for the end-to-end job indexing orchestration."""

from pathlib import Path
from uuid import uuid4

import chromadb
import pandas as pd
import pytest

from app.indexing_service import index_jobs
from app.vector_store import JobVectorStore


@pytest.fixture
def vector_store() -> JobVectorStore:
    return JobVectorStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"talentfit-index-test-{uuid4().hex}",
    )


def fake_embedder(documents: list[str]) -> list[list[float]]:
    """Return stable vectors without calling Ollama."""
    return [
        [
            float("Job Title: Java " in document),
            float("Job Title: Python " in document),
            0.5,
        ]
        for document in documents
    ]


def test_index_jobs_connects_the_full_pipeline(
    vector_store: JobVectorStore,
) -> None:
    summary = index_jobs(
        "resources/sample_jobs.csv",
        vector_store=vector_store,
        embedder=fake_embedder,
    )

    assert summary.loaded_jobs == 5
    assert summary.indexed_jobs == 5
    assert summary.inserted_jobs == 5
    assert summary.updated_jobs == 0
    assert summary.deleted_jobs == 0
    assert summary.skipped_jobs == 0
    assert summary.collection_count == 5

    hits = vector_store.search_jobs([1.0, 0.0, 0.5], limit=1)
    assert hits[0].job.job_id == "JOB-001"
    assert "Job Title: Java Backend Engineer" in hits[0].document


def test_reindex_skips_unchanged_jobs_without_embedding_again(
    vector_store: JobVectorStore,
) -> None:
    first = index_jobs(
        "resources/sample_jobs.csv",
        vector_store=vector_store,
        embedder=fake_embedder,
    )
    embedding_calls: list[list[str]] = []
    second = index_jobs(
        "resources/sample_jobs.csv",
        vector_store=vector_store,
        embedder=lambda documents: embedding_calls.append(documents) or fake_embedder(documents),
    )

    assert first.collection_count == 5
    assert second.collection_count == 5
    assert second.indexed_jobs == 0
    assert second.inserted_jobs == 0
    assert second.updated_jobs == 0
    assert second.skipped_jobs == 5
    assert embedding_calls == []


def test_reindex_updates_only_changed_job(
    vector_store: JobVectorStore, tmp_path: Path
) -> None:
    index_jobs(
        "resources/sample_jobs.csv",
        vector_store=vector_store,
        embedder=fake_embedder,
    )
    dataframe = pd.read_csv("resources/sample_jobs.csv")
    dataframe.loc[0, "title"] = "Senior Java Backend Engineer"
    changed_file = tmp_path / "changed_jobs.csv"
    dataframe.to_csv(changed_file, index=False)
    embedded_documents: list[str] = []

    summary = index_jobs(
        changed_file,
        vector_store=vector_store,
        embedder=lambda documents: embedded_documents.extend(documents)
        or fake_embedder(documents),
    )

    assert summary.inserted_jobs == 0
    assert summary.updated_jobs == 1
    assert summary.skipped_jobs == 4
    assert len(embedded_documents) == 1
    assert "Job Title: Senior Java Backend Engineer" in embedded_documents[0]
    assert vector_store.count_jobs() == 5


def test_reject_duplicate_job_ids(
    vector_store: JobVectorStore, tmp_path: Path
) -> None:
    dataframe = pd.read_csv("resources/sample_jobs.csv")
    dataframe.loc[1, "job_id"] = dataframe.loc[0, "job_id"]
    duplicate_file = tmp_path / "duplicate_jobs.csv"
    dataframe.to_csv(duplicate_file, index=False)

    with pytest.raises(ValueError, match="duplicate job_id"):
        index_jobs(
            duplicate_file,
            vector_store=vector_store,
            embedder=fake_embedder,
        )


def test_incremental_snapshot_removes_jobs_missing_from_new_file(
    vector_store: JobVectorStore,
) -> None:
    first = index_jobs(
        "resources/sample_ats_1.xlsx",
        vector_store=vector_store,
        embedder=fake_embedder,
    )
    second = index_jobs(
        "resources/sample_ats_2_incremental.xlsx",
        vector_store=vector_store,
        embedder=fake_embedder,
    )

    assert first.collection_count == 4
    assert second.loaded_jobs == 4
    assert second.inserted_jobs == 1
    assert second.updated_jobs == 1
    assert second.deleted_jobs == 1
    assert second.skipped_jobs == 2
    assert second.collection_count == 4
    assert set(vector_store.list_job_ids()) == {"18846", "18848", "18849", "18850"}


def test_reject_embedding_count_mismatch(vector_store: JobVectorStore) -> None:
    with pytest.raises(RuntimeError, match="1 vectors for 5 jobs"):
        index_jobs(
            "resources/sample_jobs.csv",
            vector_store=vector_store,
            embedder=lambda documents: [[0.1, 0.2]],
        )
