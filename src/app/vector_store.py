"""ChromaDB persistence and semantic search for TalentFit jobs."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.config import CHROMA_PATH, COLLECTION_NAME, TOP_K_RETRIEVAL
from app.models import Job


@dataclass(frozen=True, slots=True)
class JobSearchHit:
    """A job returned by vector search with its semantic similarity."""

    job: Job
    document: str
    distance: float
    semantic_score: float


class JobVectorStore:
    """Keep all Chroma-specific job storage operations behind one interface."""

    def __init__(
        self,
        *,
        path: str | Path = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        client: Any | None = None,
    ) -> None:
        self._client = client if client is not None else chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._ensure_filter_metadata()

    def _ensure_filter_metadata(self) -> None:
        """Upgrade legacy records so native status and Geo filters remain usable."""
        sample = self._collection.get(limit=1, include=["metadatas"])
        if not sample["ids"]:
            return
        sample_metadata = (sample["metadatas"] or [{}])[0] or {}
        if "status" in sample_metadata and "geo" in sample_metadata:
            return

        result = self._collection.get(include=["metadatas"])
        upgraded: list[dict[str, Any]] = []
        for metadata in result["metadatas"] or []:
            resolved = dict(metadata or {})
            job_json = resolved.get("job_json")
            if isinstance(job_json, str):
                job = Job.model_validate_json(job_json)
                resolved["status"] = job.status
                resolved["geo"] = (job.geo or "").strip().casefold()
            upgraded.append(resolved)
        if result["ids"]:
            self._collection.update(ids=result["ids"], metadatas=upgraded)

    def count_jobs(self) -> int:
        """Return the number of indexed jobs."""
        return self._collection.count()

    def clear_jobs(self) -> int:
        """Delete every indexed job and return the number removed."""
        result = self._collection.get(include=[])
        job_ids = result["ids"]
        if job_ids:
            self._collection.delete(ids=job_ids)
        return len(job_ids)

    def list_job_ids(self) -> list[str]:
        """Return every indexed ATS reference number."""
        return list(self._collection.get(include=[])["ids"])

    def delete_jobs(self, job_ids: list[str]) -> int:
        """Delete the supplied job IDs and return the number that existed."""
        if not job_ids:
            return 0
        existing_ids = list(self._collection.get(ids=job_ids, include=[])["ids"])
        if existing_ids:
            self._collection.delete(ids=existing_ids)
        return len(existing_ids)

    def upsert_job(
        self,
        job: Job,
        document: str,
        embedding: list[float],
        *,
        content_hash: str | None = None,
    ) -> None:
        """Insert a job or replace the existing job with the same ID."""
        hashes = [content_hash] if content_hash is not None else None
        self.upsert_jobs([job], [document], [embedding], content_hashes=hashes)

    def upsert_jobs(
        self,
        jobs: list[Job],
        documents: list[str],
        embeddings: list[list[float]],
        *,
        content_hashes: list[str] | None = None,
    ) -> None:
        """Insert or update a batch of jobs and their precomputed embeddings."""
        if not (len(jobs) == len(documents) == len(embeddings)):
            raise ValueError("jobs, documents, and embeddings must have equal lengths")
        if content_hashes is not None and len(content_hashes) != len(jobs):
            raise ValueError("content_hashes must have the same length as jobs")
        if not jobs:
            return
        if any(not document.strip() for document in documents):
            raise ValueError("Job documents must not be empty")
        if any(not embedding for embedding in embeddings):
            raise ValueError("Job embeddings must not be empty")

        resolved_hashes = content_hashes or [
            hashlib.sha256(
                f"{job.model_dump_json()}\n{document}".encode("utf-8")
            ).hexdigest()
            for job, document in zip(jobs, documents, strict=True)
        ]

        self._collection.upsert(
            ids=[job.job_id for job in jobs],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "job_json": job.model_dump_json(),
                    "content_hash": content_hash,
                    "status": job.status,
                    "geo": (job.geo or "").strip().casefold(),
                }
                for job, content_hash in zip(jobs, resolved_hashes, strict=True)
            ],
        )

    def get_content_hashes(self, job_ids: list[str]) -> dict[str, str]:
        """Return stored content hashes for existing job IDs."""
        if not job_ids:
            return {}

        result = self._collection.get(ids=job_ids, include=["metadatas"])
        hashes: dict[str, str] = {}
        for job_id, metadata in zip(
            result["ids"], result["metadatas"] or [], strict=True
        ):
            stored_hash = metadata.get("content_hash") if metadata is not None else None
            hashes[job_id] = stored_hash if isinstance(stored_hash, str) else ""
        return hashes

    @staticmethod
    def _job_filter(
        *, open_only: bool, india_only: bool = False
    ) -> dict[str, Any] | None:
        filters: list[dict[str, Any]] = []
        if open_only:
            filters.append({"status": "open"})
        if india_only:
            filters.append({"geo": "india"})
        if len(filters) == 1:
            return filters[0]
        if filters:
            return {"$and": filters}
        return None

    def get_jobs(
        self, *, open_only: bool = False, india_only: bool = False
    ) -> list[Job]:
        """Return indexed jobs matching recruiter-facing availability filters."""
        result = self._collection.get(
            where=self._job_filter(open_only=open_only, india_only=india_only),
            include=["metadatas"],
        )
        jobs = [
            Job.model_validate_json(metadata["job_json"])
            for metadata in result["metadatas"] or []
            if metadata is not None and isinstance(metadata.get("job_json"), str)
        ]
        return sorted(jobs, key=lambda job: job.job_id)

    def search_jobs(
        self,
        query_embedding: list[float],
        *,
        limit: int = TOP_K_RETRIEVAL,
        open_only: bool = False,
        india_only: bool = False,
    ) -> list[JobSearchHit]:
        """Return jobs nearest to a precomputed query embedding."""
        if not query_embedding:
            raise ValueError("Query embedding must not be empty")
        if limit <= 0:
            raise ValueError("Search limit must be greater than zero")
        if self.count_jobs() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, self.count_jobs()),
            where=self._job_filter(open_only=open_only, india_only=india_only),
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        hits: list[JobSearchHit] = []
        for job_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            if document is None or metadata is None or distance is None:
                raise RuntimeError(f"Chroma returned incomplete data for job {job_id}")

            job = Job.model_validate_json(metadata["job_json"])
            numeric_distance = float(distance)
            similarity = max(0.0, min(1.0, 1.0 - numeric_distance))
            hits.append(
                JobSearchHit(
                    job=job,
                    document=document,
                    distance=numeric_distance,
                    semantic_score=similarity,
                )
            )
            if len(hits) == limit:
                break

        return hits
