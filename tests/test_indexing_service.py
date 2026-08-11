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


@pytest.fixture
def india_jobs_file(tmp_path: Path) -> Path:
    """Copy the compact legacy fixture with an explicit eligible Geo."""
    dataframe = pd.read_csv("resources/sample_jobs.csv")
    dataframe["Geo"] = "India"
    file_path = tmp_path / "india_jobs.csv"
    dataframe.to_csv(file_path, index=False)
    return file_path


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
    india_jobs_file: Path,
) -> None:
    summary = index_jobs(
        india_jobs_file,
        vector_store=vector_store,
        embedder=fake_embedder,
    )

    assert summary.loaded_jobs == 5
    assert summary.eligible_jobs == 5
    assert summary.excluded_jobs == 0
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
    india_jobs_file: Path,
) -> None:
    first = index_jobs(
        india_jobs_file,
        vector_store=vector_store,
        embedder=fake_embedder,
    )
    embedding_calls: list[list[str]] = []
    second = index_jobs(
        india_jobs_file,
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
    vector_store: JobVectorStore, tmp_path: Path, india_jobs_file: Path
) -> None:
    index_jobs(
        india_jobs_file,
        vector_store=vector_store,
        embedder=fake_embedder,
    )
    dataframe = pd.read_csv(india_jobs_file)
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
    dataframe["Geo"] = "India"
    dataframe.loc[1, "job_id"] = dataframe.loc[0, "job_id"]
    duplicate_file = tmp_path / "duplicate_jobs.csv"
    dataframe.to_csv(duplicate_file, index=False)

    with pytest.raises(ValueError, match="duplicate job_id"):
        index_jobs(
            duplicate_file,
            vector_store=vector_store,
            embedder=fake_embedder,
        )


def test_incremental_snapshot_synchronizes_only_india_jobs(
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

    assert first.loaded_jobs == 4
    assert first.eligible_jobs == 3
    assert first.excluded_jobs == 1
    assert first.collection_count == 3
    assert second.loaded_jobs == 4
    assert second.eligible_jobs == 3
    assert second.excluded_jobs == 1
    assert second.inserted_jobs == 0
    assert second.updated_jobs == 1
    assert second.deleted_jobs == 0
    assert second.skipped_jobs == 2
    assert second.collection_count == 3
    assert set(vector_store.list_job_ids()) == {"18846", "18848", "18849"}


def test_reject_embedding_count_mismatch(
    vector_store: JobVectorStore, india_jobs_file: Path
) -> None:
    with pytest.raises(RuntimeError, match="1 vectors for 5 jobs"):
        index_jobs(
            india_jobs_file,
            vector_store=vector_store,
            embedder=lambda documents: [[0.1, 0.2]],
        )


def test_non_india_jobs_are_not_embedded_or_indexed(
    vector_store: JobVectorStore, tmp_path: Path
) -> None:
    file_path = tmp_path / "mixed_geo.csv"
    pd.DataFrame(
        [
            {"Reference Number": "INDIA", "Job Title": "India job", "Geo": "India"},
            {"Reference Number": "USA", "Job Title": "USA job", "Geo": "USA"},
            {"Reference Number": "MISSING", "Job Title": "No Geo"},
        ]
    ).to_csv(file_path, index=False)
    embedded_documents: list[str] = []

    summary = index_jobs(
        file_path,
        vector_store=vector_store,
        embedder=lambda documents: embedded_documents.extend(documents)
        or fake_embedder(documents),
    )

    assert summary.loaded_jobs == 3
    assert summary.eligible_jobs == 1
    assert summary.excluded_jobs == 2
    assert vector_store.list_job_ids() == ["INDIA"]
    assert len(embedded_documents) == 1
