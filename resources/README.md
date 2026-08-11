# TalentFit Sample Resources

These files are reusable development fixtures:

- `sample_ats_1.xlsx` - baseline ATS snapshot with four jobs and extra ignored columns.
- `sample_ats_2_incremental.xlsx` - next snapshot: one job is updated, one is added,
  one is unchanged, and one baseline job is removed from the index.
- `sample_ats_3_100_jobs.xlsx` - larger ATS snapshot with 100 varied job records for
  upload, indexing, search, and performance checks.
- `sample_ats_4_500_jobs.xlsx` - independent 500-job ATS snapshot with different
  references, technologies, geographies, and statuses.
- `sample_jobs.csv` - legacy compact CSV fixture used by automated tests.
- `sample_resume.pdf` - a text-based resume for PDF extraction and end-to-end matching.
- `sample_resume_data_ai_engineer.pdf` - chronological data and AI engineering CV.
- `sample_resume_cloud_sre.pdf` - project-focused cloud platform and SRE CV.

Example commands:

```bash
streamlit run
```

Upload `sample_ats_1.xlsx`, then `sample_ats_2_incremental.xlsx` in the ATS upload
section. Upload `sample_resume.pdf` in the resume match section afterward.

TalentFit requires only `Reference Number` and `Job Title`. It recognizes and
indexes the supported recruiter fields, tolerates common header variants, and ignores
all unrelated columns. Each upload is treated as the current ATS snapshot, so jobs
missing from a later upload are removed from ChromaDB.
