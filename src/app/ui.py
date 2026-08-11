"""Streamlit interface for the TalentFit MVP."""

import csv
import hashlib
import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import pandas as pd
import streamlit as st

from app.ats_service import ats_preview, read_ats_file
from app.config import EMBEDDING_MODEL, OLLAMA_MODEL
from app.document_builder import format_experience
from app.indexing_service import index_jobs
from app.models import Job, MatchResult
from app.resume_matching_service import (
    PreparedResume,
    ResumeMatchSummary,
    match_prepared_resume,
    prepare_resume,
)
from app.vector_store import JobVectorStore


class UploadedFile(Protocol):
    """Minimal uploaded-file interface required by this UI."""

    name: str

    def getvalue(self) -> bytes: ...


def persist_upload(upload: UploadedFile, directory: str | Path) -> Path:
    """Store an uploaded file under a safe generated name."""
    suffix = Path(upload.name).suffix.casefold()
    destination = Path(directory) / f"upload{suffix}"
    destination.write_bytes(upload.getvalue())
    return destination


@st.cache_data(max_entries=10, show_spinner=False)
def load_ats_preview(content: bytes, suffix: str) -> pd.DataFrame:
    """Read and validate an ATS upload for preview without indexing it."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / f"preview{suffix.casefold()}"
        path.write_bytes(content)
        return ats_preview(read_ats_file(path))


def build_match_report_csv(summary: ResumeMatchSummary) -> bytes:
    """Create a recruiter-friendly CSV report from the current match results."""
    output = StringIO()
    fieldnames = [
        "candidate_name",
        "candidate_experience_years",
        "candidate_location",
        "candidate_roles",
        "candidate_skills",
        "rank",
        "job_id",
        "job_created_date",
        "job_title",
        "open_positions",
        "designation",
        "geo",
        "business_unit",
        "min_experience",
        "max_experience",
        "mandatory_skills",
        "job_status",
        "referral_enabled",
        "final_score",
        "mandatory_score",
        "experience_score",
        "semantic_score",
        "matched_mandatory",
        "missing_mandatory",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    candidate = summary.candidate
    for rank, match in enumerate(summary.matches, start=1):
        writer.writerow(
            {
                "candidate_name": candidate.name,
                "candidate_experience_years": candidate.experience_years,
                "candidate_location": candidate.location or "",
                "candidate_roles": "; ".join(candidate.roles),
                "candidate_skills": "; ".join(candidate.skills),
                "rank": rank,
                "job_id": match.job.job_id,
                "job_created_date": match.job.created_date or "",
                "job_title": match.job.title,
                "open_positions": match.job.open_positions or 0,
                "designation": match.job.designation or "",
                "geo": match.job.geo or "",
                "business_unit": match.job.business_unit or "",
                "min_experience": match.job.min_experience_years or "",
                "max_experience": match.job.max_experience_years or "",
                "mandatory_skills": "; ".join(match.job.mandatory_skills),
                "job_status": match.job.status,
                "referral_enabled": "Yes" if match.job.referral_allowed else "No",
                "final_score": match.final_score,
                "mandatory_score": match.mandatory_score,
                "experience_score": match.experience_score,
                "semantic_score": round(match.semantic_score * 100, 2),
                "matched_mandatory": "; ".join(match.matched_mandatory),
                "missing_mandatory": "; ".join(match.missing_mandatory),
            }
        )
    return output.getvalue().encode("utf-8")


def _report_filename(candidate_name: str) -> str:
    safe_name = re.sub(r"[^a-z0-9]+", "_", candidate_name.casefold()).strip("_")
    return f"talentfit_matches_{safe_name or 'candidate'}.csv"


def _skill_text(skills: list[str]) -> str:
    return ", ".join(skills) if skills else "None"


def _render_match(position: int, match: MatchResult) -> None:
    with st.container(border=True):
        title_column, score_column = st.columns([4, 1])
        title_column.subheader(f"{position}. {match.job.title}")
        title_column.caption(
            " · ".join(
                [
                    f"Reference: {match.job.job_id}",
                    f"Created: {match.job.created_date or 'Not specified'}",
                    f"Status: {match.job.status}",
                    f"Geo: {match.job.geo or 'Not specified'}",
                    f"Experience: {format_experience(match.job)}",
                ]
            )
        )
        score_column.metric("Match", f"{match.final_score:.1f}%")
        st.progress(match.final_score / 100)

        details = st.columns(3)
        details[0].metric("Open positions", match.job.open_positions or 0)
        details[1].metric("Designation", match.job.designation or "Not specified")
        details[2].metric("Business unit", match.job.business_unit or "Not specified")

        score_columns = st.columns(3)
        score_columns[0].metric("Mandatory", f"{match.mandatory_score:.0f}%")
        score_columns[1].metric("Experience", f"{match.experience_score:.0f}%")
        score_columns[2].metric("Semantic", f"{match.semantic_score * 100:.0f}%")

        matched_column, missing_column = st.columns(2)
        matched_column.markdown("**Matched mandatory skills**")
        matched_column.write(_skill_text(match.matched_mandatory))
        missing_column.markdown("**Missing mandatory skills**")
        missing_column.write(_skill_text(match.missing_mandatory))

        st.caption(
            f"Referral enabled: {'Yes' if match.job.referral_allowed else 'No'}"
        )


def _render_candidate(summary: ResumeMatchSummary) -> None:
    candidate = summary.candidate
    st.subheader("Extracted candidate profile")
    profile_columns = st.columns(3)
    profile_columns[0].metric("Name", candidate.name)
    profile_columns[1].metric("Experience", f"{candidate.experience_years:g} years")
    profile_columns[2].metric("Location", candidate.location or "Not specified")
    st.markdown(f"**Roles:** {_skill_text(candidate.roles)}")
    st.markdown(f"**Skills:** {_skill_text(candidate.skills)}")
    if candidate.summary:
        st.markdown(f"**Summary:** {candidate.summary}")


def build_job_dashboard_dataframe(jobs: list[Job]) -> pd.DataFrame:
    """Convert indexed jobs into the supported recruiter-facing table."""
    dataframe = pd.DataFrame(
        [
            {
                "Reference Number": job.job_id,
                "Job Created Date": job.created_date,
                "Job Title": job.title,
                "Open Positions": job.open_positions or 0,
                "Designation": job.designation or "",
                "Geo": job.geo or "",
                "Business Unit": job.business_unit or "",
                "Min Experience": job.min_experience_years,
                "Max Experience": job.max_experience_years,
                "Mandatory Skills": ", ".join(job.mandatory_skills),
                "Job Status": job.status,
                "Referral Enabled": job.referral_allowed,
            }
            for job in jobs
        ]
    )
    if not dataframe.empty:
        dataframe["Job Created Date"] = pd.to_datetime(
            dataframe["Job Created Date"], errors="coerce"
        )
    return dataframe


def _clear_ats_upload() -> None:
    st.session_state["ats_upload_version"] += 1
    st.session_state.pop("last_index_summary", None)


def _clear_resume_workflow() -> None:
    st.session_state["resume_upload_version"] += 1
    st.session_state.pop("resume_match_summary", None)
    st.session_state["resume_cache"] = {}


def _clear_resume_results() -> None:
    """Discard results when search criteria change."""
    st.session_state.pop("resume_match_summary", None)


def _resume_cache_key(content: bytes) -> str:
    """Bind cached analysis to both file content and configured models."""
    digest = hashlib.sha256(content).hexdigest()
    return f"{digest}:{OLLAMA_MODEL}:{EMBEDDING_MODEL}"


def _remember_prepared_resume(key: str, prepared: PreparedResume) -> None:
    """Keep a small session-only cache of recently analyzed resumes."""
    cache: dict[str, PreparedResume] = st.session_state["resume_cache"]
    cache[key] = prepared
    while len(cache) > 5:
        cache.pop(next(iter(cache)))


@st.dialog("Clear indexed jobs?", icon=":material/delete:")
def _confirm_clear_index() -> None:
    count = JobVectorStore().count_jobs()
    st.warning(
        f"This will remove all {count} indexed jobs. You can restore them by uploading "
        "the ATS file again."
    )
    if st.button(
        "Clear job index",
        type="primary",
        icon=":material/delete_forever:",
        disabled=count == 0,
    ):
        removed = JobVectorStore().clear_jobs()
        st.session_state.pop("last_index_summary", None)
        st.session_state.pop("resume_match_summary", None)
        st.toast(f"Removed {removed} indexed jobs", icon=":material/check_circle:")
        st.rerun()


def render_ats_upload() -> None:
    """Render ATS upload and indexing controls."""
    st.header("ATS Upload")
    st.write("Upload a CSV or Excel export to clean, embed, and index its jobs.")

    try:
        st.metric("Currently indexed jobs", JobVectorStore().count_jobs())
    except Exception as exc:  # Streamlit should display storage startup failures.
        st.error(f"Could not open the job collection: {exc}")

    upload = st.file_uploader(
        "ATS file",
        type=["csv", "xlsx", "xls"],
        key=f"ats_upload_{st.session_state['ats_upload_version']}",
    )

    preview: pd.DataFrame | None = None
    if upload is not None:
        try:
            preview = load_ats_preview(upload.getvalue(), Path(upload.name).suffix)
            st.subheader("Upload preview")
            preview_metrics = st.columns(2)
            preview_metrics[0].metric("Rows", len(preview))
            preview_metrics[1].metric("Columns", len(preview.columns))
            st.dataframe(preview.head(10), hide_index=True, width="stretch")
            st.caption(
                f"Showing the first {min(10, len(preview))} rows. "
                f"Detected columns: {', '.join(map(str, preview.columns))}"
            )
        except Exception as exc:
            st.error(f"ATS preview failed: {exc}", icon=":material/error:")

    with st.container(horizontal=True):
        process_clicked = st.button(
            "Confirm and index",
            type="primary",
            icon=":material/database_upload:",
            disabled=preview is None,
        )
        if st.button(
            "Clear upload",
            icon=":material/restart_alt:",
            disabled=upload is None,
        ):
            _clear_ats_upload()
            st.rerun()
        if st.button("Clear job index", icon=":material/delete:"):
            _confirm_clear_index()

    if process_clicked:
        assert upload is not None
        try:
            with st.status("Processing ATS jobs...", expanded=True) as status:
                st.write("Cleaning and validating jobs")
                with TemporaryDirectory() as directory:
                    file_path = persist_upload(upload, directory)
                    st.write("Generating embeddings and updating the index")
                    summary = index_jobs(file_path)
                status.update(label="ATS indexing complete", state="complete", expanded=False)
            st.session_state["last_index_summary"] = summary
        except Exception as exc:
            st.error(f"ATS processing failed: {exc}", icon=":material/error:")

    summary = st.session_state.get("last_index_summary")
    if summary is not None:
        st.success(
                f"Processed {summary.loaded_jobs} jobs: "
                f"{summary.inserted_jobs} inserted, {summary.updated_jobs} updated, "
                f"{summary.deleted_jobs} removed, "
                f"and {summary.skipped_jobs} unchanged. "
                f"Collection contains {summary.collection_count} jobs.",
                icon=":material/check_circle:",
        )


def render_resume_match() -> None:
    """Render resume analysis and explainable matching controls."""
    st.header("Resume Match")
    st.write("Upload a text-based PDF resume to find its top three job matches.")

    upload = st.file_uploader(
        "Resume PDF",
        type=["pdf"],
        key=f"resume_upload_{st.session_state['resume_upload_version']}",
    )
    st.subheader("Job filters")
    st.caption("Filters are applied before the top matches are ranked.")
    with st.container(horizontal=True):
        open_only = st.toggle(
            "Open jobs only",
            value=True,
            key="resume_filter_open_only",
            help="Exclude closed, rejected, draft, and on-hold jobs.",
            on_change=_clear_resume_results,
            persist_state="session",
        )
        referral_only = st.toggle(
            "Referral-enabled only",
            value=False,
            key="resume_filter_referral_only",
            help="Include only jobs that allow employee referrals.",
            on_change=_clear_resume_results,
            persist_state="session",
        )
    with st.container(horizontal=True):
        analyze_clicked = st.button(
            "Analyze resume",
            type="primary",
            icon=":material/analytics:",
            disabled=upload is None,
        )
        if st.button(
            "Start over",
            icon=":material/restart_alt:",
            disabled=upload is None and "resume_match_summary" not in st.session_state,
        ):
            _clear_resume_workflow()
            st.rerun()

    if analyze_clicked:
        assert upload is not None
        try:
            with st.status("Analyzing resume...", expanded=True) as status:
                content = upload.getvalue()
                cache_key = _resume_cache_key(content)
                prepared = st.session_state["resume_cache"].get(cache_key)
                cache_hit = prepared is not None
                if prepared is None:
                    st.write("Extracting text, candidate details, and embedding")
                    with TemporaryDirectory() as directory:
                        file_path = persist_upload(upload, directory)
                        prepared = prepare_resume(file_path)
                    _remember_prepared_resume(cache_key, prepared)
                else:
                    st.write("Reusing cached candidate analysis and embedding")

                st.write("Searching and ranking eligible indexed jobs")
                st.session_state["resume_match_summary"] = match_prepared_resume(
                    prepared,
                    open_only=open_only,
                    referral_only=referral_only,
                    cache_hit=cache_hit,
                )
                status.update(label="Resume analysis complete", state="complete", expanded=False)
        except Exception as exc:  # Surface extraction/model failures in the MVP UI.
            st.session_state.pop("resume_match_summary", None)
            st.error(f"Resume analysis failed: {exc}", icon=":material/error:")

    summary = st.session_state.get("resume_match_summary")
    if summary is None:
        return

    _render_candidate(summary)
    timing_text = " · ".join(
        f"{label.replace('_', ' ').title()}: {seconds:.2f}s"
        for label, seconds in summary.timings.items()
    )
    if summary.cache_hit:
        timing_text = f"Candidate cache reused · {timing_text}"
    if timing_text:
        st.caption(timing_text)
    st.subheader("Top job matches")
    if not summary.matches:
        st.info(
            "No indexed jobs matched the selected filters. Try relaxing a filter or "
            "uploading a newer ATS snapshot."
        )
        return
    for position, match in enumerate(summary.matches, start=1):
        _render_match(position, match)

    st.download_button(
        "Download match report",
        data=build_match_report_csv(summary),
        file_name=_report_filename(summary.candidate.name),
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
    )


def render_job_dashboard() -> None:
    """Render an availability dashboard for jobs currently indexed in Chroma."""
    st.header("Job dashboard")
    st.write("Browse the current ATS snapshot stored in the TalentFit job index.")

    st.subheader("Job filters")
    st.caption("Filters are applied before the available jobs are displayed.")
    with st.container(horizontal=True):
        open_only = st.toggle(
            "Open jobs only",
            value=True,
            key="dashboard_filter_open_only",
            help="Exclude closed, rejected, draft, and on-hold jobs.",
            persist_state="session",
        )
        referral_only = st.toggle(
            "Referral-enabled only",
            value=False,
            key="dashboard_filter_referral_only",
            help="Include only jobs that allow employee referrals.",
            persist_state="session",
        )

    try:
        store = JobVectorStore()
        indexed_count = store.count_jobs()
        jobs = store.get_jobs(
            open_only=open_only,
            referral_only=referral_only,
        )
    except Exception as exc:
        st.error(f"Could not read the job collection: {exc}", icon=":material/error:")
        return

    open_position_count = sum(job.open_positions or 0 for job in jobs)
    referral_count = sum(job.referral_allowed for job in jobs)
    with st.container(horizontal=True):
        st.metric("Indexed jobs", indexed_count, border=True)
        st.metric("Matching jobs", len(jobs), border=True)
        st.metric("Open positions", open_position_count, border=True)
        st.metric("Referral-enabled", referral_count, border=True)

    if not jobs:
        st.info(
            "No indexed jobs matched the selected filters. Try relaxing a filter or "
            "uploading a newer ATS snapshot."
        )
        return

    st.subheader("Available jobs")
    st.caption(f"Showing {len(jobs)} jobs from the current indexed ATS snapshot.")
    st.dataframe(
        build_job_dashboard_dataframe(jobs),
        hide_index=True,
        height=600,
        key="indexed_jobs_dashboard",
        column_config={
            "Reference Number": st.column_config.TextColumn(pinned=True),
            "Job Created Date": st.column_config.DateColumn(format="DD-MMM-YYYY"),
            "Job Title": st.column_config.TextColumn(pinned=True),
            "Open Positions": st.column_config.NumberColumn(format="%d"),
            "Min Experience": st.column_config.NumberColumn(format="%.1f years"),
            "Max Experience": st.column_config.NumberColumn(format="%.1f years"),
            "Referral Enabled": st.column_config.CheckboxColumn(),
        },
    )


def run_app() -> None:
    """Run the TalentFit Streamlit application."""
    st.set_page_config(
        page_title="TalentFit", page_icon=":material/work:", layout="wide"
    )
    st.session_state.setdefault("ats_upload_version", 0)
    st.session_state.setdefault("resume_upload_version", 0)
    st.session_state.setdefault("resume_cache", {})
    st.title("TalentFit")
    st.caption("Explainable semantic matching between candidates and open roles")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a section",
            ["ATS Upload", "Job Dashboard", "Resume Match"],
        )
        st.caption(f"Chat model: {OLLAMA_MODEL}")
        st.caption(f"Embedding model: {EMBEDDING_MODEL}")

    if page == "ATS Upload":
        render_ats_upload()
    elif page == "Job Dashboard":
        render_job_dashboard()
    else:
        render_resume_match()
