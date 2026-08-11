"""Orchestrate ATS loading, document creation, embedding, and indexing."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.ats_service import load_jobs
from app.document_builder import build_job_documents
from app.embedding_service import embed_texts
from app.vector_store import JobVectorStore


@dataclass(frozen=True, slots=True)
class IndexingSummary:
    """Summary of a completed ATS indexing operation."""

    loaded_jobs: int
    eligible_jobs: int
    excluded_jobs: int
    indexed_jobs: int
    inserted_jobs: int
    updated_jobs: int
    deleted_jobs: int
    skipped_jobs: int
    collection_count: int


EmbeddingFunction = Callable[[list[str]], list[list[float]]]
INDEX_SCHEMA_VERSION = "5"


def build_content_hash(job_json: str, document: str) -> str:
    """Build a stable hash for all normalized job content."""
    content = f"{INDEX_SCHEMA_VERSION}\n{job_json}\n{document}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def index_jobs(
    file_path: str | Path,
    *,
    vector_store: JobVectorStore | None = None,
    embedder: EmbeddingFunction = embed_texts,
) -> IndexingSummary:
    """Load an ATS snapshot and index only jobs whose Geo is India."""
    loaded_jobs = load_jobs(file_path)
    jobs = [
        job
        for job in loaded_jobs
        if (job.geo or "").strip().casefold() == "india"
    ]
    documents = build_job_documents(jobs)
    job_ids = [job.job_id for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("ATS file contains duplicate job_id values")

    content_hashes = [
        build_content_hash(job.model_dump_json(), document)
        for job, document in zip(jobs, documents, strict=True)
    ]

    store = vector_store or JobVectorStore()
    existing_job_ids = set(store.list_job_ids())
    existing_hashes = store.get_content_hashes(job_ids)

    insert_indexes: list[int] = []
    update_indexes: list[int] = []
    for index, (job_id, content_hash) in enumerate(
        zip(job_ids, content_hashes, strict=True)
    ):
        if job_id not in existing_hashes:
            insert_indexes.append(index)
        elif existing_hashes[job_id] != content_hash:
            update_indexes.append(index)

    index_positions = insert_indexes + update_indexes
    jobs_to_index = [jobs[index] for index in index_positions]
    documents_to_index = [documents[index] for index in index_positions]
    hashes_to_index = [content_hashes[index] for index in index_positions]
    embeddings = embedder(documents_to_index) if documents_to_index else []

    if len(embeddings) != len(jobs_to_index):
        raise RuntimeError(
            f"Embedding service returned {len(embeddings)} vectors for "
            f"{len(jobs_to_index)} jobs"
        )

    store.upsert_jobs(
        jobs_to_index,
        documents_to_index,
        embeddings,
        content_hashes=hashes_to_index,
    )
    deleted_jobs = store.delete_jobs(sorted(existing_job_ids.difference(job_ids)))

    return IndexingSummary(
        loaded_jobs=len(loaded_jobs),
        eligible_jobs=len(jobs),
        excluded_jobs=len(loaded_jobs) - len(jobs),
        indexed_jobs=len(jobs_to_index),
        inserted_jobs=len(insert_indexes),
        updated_jobs=len(update_indexes),
        deleted_jobs=deleted_jobs,
        skipped_jobs=len(jobs) - len(jobs_to_index),
        collection_count=store.count_jobs(),
    )
