# TalentFit MVP

TalentFit is a local, explainable recruitment-matching application built with
Streamlit, Ollama, and ChromaDB. It converts an ATS job export into a searchable job
index, extracts a structured candidate profile from a resume, and returns the strongest
job matches with transparent skill gaps and scores.

By default, job data, resume text, model inference, embeddings, and the vector database
stay on the machine running the application. No cloud API key is required.

## What the application can do

- Upload ATS exports in CSV, XLS, or XLSX format.
- Normalize supported column names and ignore unrelated columns.
- Preview normalized ATS data before changing the index.
- Insert new jobs, update changed jobs, skip unchanged jobs, and remove missing jobs.
- Browse all indexed jobs from a recruiter-facing dashboard.
- Filter dashboard and resume results by open status and referral availability.
- Read differently structured, multi-page, text-based PDF resumes.
- Use Qwen to extract a validated candidate profile.
- Verify professional experience using explicit totals and non-overlapping work dates.
- Retrieve semantically similar jobs using Ollama embeddings and ChromaDB.
- Rank results with deterministic skill, experience, and semantic scoring.
- Show the top three jobs with matched and missing mandatory skills.
- Export candidate and matching results as a CSV report.
- Cache prepared resumes during the current session for faster repeated searches.
- Clear uploads, resume results, or the complete local job index from the UI.

## Application pages

### ATS Upload

Use this page to load the current ATS job snapshot.

1. Upload a CSV, XLS, or XLSX file.
2. Review the normalized row count, column count, and first ten rows.
3. Select **Confirm and index**.
4. Review the inserted, updated, unchanged, removed, and total job counts.

Available actions:

- **Clear upload** removes the selected file from the page without changing indexed
  jobs.
- **Clear job index** asks for confirmation and permanently removes all locally indexed
  jobs. The index can be restored by uploading the ATS snapshot again.

### Job Dashboard

Use this page to inspect the jobs currently stored in ChromaDB without uploading a
resume.

The dashboard shows:

- Total indexed jobs
- Jobs matching the active filters
- Total open positions across the filtered jobs
- Number of referral-enabled filtered jobs
- A table containing the 12 supported recruiter-facing fields

**Open jobs only** is enabled by default. **Referral-enabled only** is disabled by
default. Both filters are applied directly through ChromaDB metadata.

### Resume Match

Use this page after at least one ATS snapshot has been indexed.

1. Upload a text-based PDF resume.
2. Keep or change the job filters.
3. Select **Analyze resume**.
4. Review the extracted candidate profile and processing timings.
5. Review the top job cards and mandatory-skill gaps.
6. Download the CSV match report if required.

The candidate profile contains the name, verified experience, location, roles, skills,
and a factual professional summary. **Start over** clears the uploaded resume, displayed
results, and the current session's prepared-resume cache.

## End-to-end architecture

```text
ATS snapshot
  -> read CSV or Excel
  -> normalize supported columns and values
  -> validate Job models
  -> build stable searchable documents
  -> create embeddings with Ollama
  -> insert/update/delete ChromaDB records

Resume PDF
  -> extract selectable text with PyMuPDF
  -> extract structured data with Qwen
  -> validate CandidateProfile with Pydantic
  -> reconcile professional experience
  -> create candidate embedding
  -> apply job filters in ChromaDB
  -> retrieve the top semantic candidates
  -> calculate deterministic match scores
  -> return and export the top results
```

## ATS data contract

TalentFit keeps and indexes only the following fields:

| Field | Required | Notes |
|---|:---:|---|
| Reference Number | Yes | Unique job identifier used for insert, update, and deletion decisions |
| Job Created Date | No | Retained as recruiter-facing search context |
| Job Title | Yes | Primary job title |
| Open Positions | No | Parsed as a whole number |
| Designation | No | Role level or internal designation |
| Geo | No | Job geography or location |
| Business Unit | No | Hiring business unit |
| Min Experience | No | Minimum years of experience |
| Max Experience | No | Maximum years of experience |
| Mandatory Skills | No | Split on commas, semicolons, or pipes and then normalized |
| Job Status | No | Normalized for filtering |
| Referral Enabled | No | Normalized to true or false |

Only **Reference Number** and **Job Title** are mandatory. Extra spreadsheet columns are
ignored and do not break the upload. Missing optional columns are created with empty
values.

### Recognized header variants

Headers are case-insensitive, and spaces or punctuation are normalized. Common variants
include:

| Canonical field | Accepted examples |
|---|---|
| Reference Number | `Reference Number`, `job_id`, `job_code` |
| Job Title | `Job Title`, `title` |
| Job Created Date | `Job Created Date`, `created_date` |
| Open Positions | `Open Position`, `Open Positions` |
| Min Experience | `Min Exp`, `Min Experience`, `min_experience_years` |
| Max Experience | `Max Exp`, `Max Experience`, `max_experience_years` |
| Job Status | `Job Status`, `status` |
| Referral Enabled | `Referral Enabled`, `Refferal Enabled`, `referral_allowed` |

If two spreadsheet columns map to the same supported field, the first one is used.
Duplicate reference numbers inside the same upload are rejected.

### Value normalization

- Skills are lowercased, deduplicated, and normalized through known aliases, such as
  `SpringBoot` to `spring boot`, `PowerBi` to `power bi`, `K8s` to `kubernetes`, and
  `Amazon Web Services` to `aws`.
- Statuses such as `active` and `published` become `open`; `filled`, `inactive`, and
  `archived` become `closed`; `hold` and `paused` become `on hold`.
- Referral values such as `1`, `true`, `yes`, and `y` become true. Unknown or missing
  referral values default to false.
- Missing spreadsheet values are cleaned instead of being stored as `NaN` text.

## Full-snapshot and incremental synchronization

Every ATS upload represents the **complete current snapshot**, not a partial patch.
TalentFit creates a stable content hash from each normalized job and its searchable
document.

```text
Reference number is new              -> INSERT and create an embedding
Reference number exists and changed  -> UPDATE and create a new embedding
Reference number exists unchanged    -> SKIP and reuse the stored embedding
Previously indexed ID is absent      -> DELETE from ChromaDB
```

This keeps the index aligned with the latest ATS export and avoids recomputing unchanged
jobs. Be careful when uploading a small file after a large one: every previously indexed
job missing from the small file will be removed.

## What is indexed

For each job, TalentFit builds a consistent text document from only the 12 supported ATS
fields. Unrelated spreadsheet columns and job-description fields are not included.

Example:

```text
Reference Number: TF-1001
Job Created Date: 12-Jul-2026
Job Title: Java Backend Engineer
Open Positions: 2
Designation: Senior Engineer
Geo: Bengaluru
Business Unit: Digital Engineering
Mandatory Skills: java, spring boot, kafka, aws
Experience: 5 to 8 years
Job Status: open
Referral Enabled: Yes
```

Ollama converts this document into an embedding. ChromaDB stores the document, embedding,
validated job JSON, content hash, status, and referral flag. The configured Chroma path
persists across application restarts.

## Resume extraction and experience verification

TalentFit uses PyMuPDF to extract selectable text from every page. It then asks the
configured Qwen chat model for structured JSON containing:

- Name
- Experience years
- Skills
- Roles
- Location
- Summary
- Professional employment periods used as verification evidence

The extractor reads the resume semantically rather than expecting a fixed template,
heading order, or layout. Pydantic validates the model output before matching begins.

Experience is reconciled deterministically:

1. Prefer a clearly stated total such as “11+ years of experience.”
2. Otherwise calculate unique professional months from employment periods.
3. Do not double-count overlapping roles.
4. Exclude education and non-professional date ranges.
5. Use the model's value only when stronger evidence is unavailable.

This guardrail prevents education dates from incorrectly inflating a candidate's total
experience.

### Supported resume formats

- Text-based PDF
- Multi-page PDF
- Different headings, section orders, prose styles, lists, and text layouts

Not currently supported:

- Scanned or image-only PDFs without a selectable text layer
- Password-protected PDFs
- Corrupted PDFs
- DOC, DOCX, image, or plain-text resume uploads through the UI

## Semantic retrieval

The candidate embedding is built from the extracted roles, skills, verified experience,
location, and summary. ChromaDB compares it with indexed job embeddings using cosine
distance.

Filters are applied **before** retrieval:

- **Open jobs only** includes jobs whose normalized status is exactly `open`.
- **Referral-enabled only** includes jobs whose referral flag is true.
- Enabling both requires both conditions.

TalentFit retrieves up to `TOP_K_RETRIEVAL` eligible jobs, then applies deterministic
matching and returns up to `TOP_K_RESULTS` final jobs. The defaults are 10 retrieved and
3 displayed.

## Understanding the scores

Semantic retrieval finds potentially relevant jobs; deterministic scoring explains and
ranks them.

| Component | Weight | What it measures |
|---|---:|---|
| Mandatory skills | 50% | Percentage of normalized mandatory job skills present in the candidate profile |
| Optional skills | 15% | Percentage of optional skills present; the current ATS contract has no optional-skill column, so jobs receive this component in full |
| Experience | 15% | Fit against the job's minimum and maximum experience range |
| Semantic similarity | 20% | Overall contextual similarity between the candidate and job embeddings |

The formula is:

```text
Final score =
    mandatory score × 0.50
  + optional score  × 0.15
  + experience score × 0.15
  + semantic score  × 0.20
```

### What “Semantic 64%” means

It means the embedding model found moderate contextual similarity between the candidate
profile and the job document. It can recognize related wording—for example, “Java
microservices lead” and “Backend Engineering Lead”—even when the titles are not exact.

It does **not** mean:

- 64% of mandatory skills matched
- A 64% chance of being hired
- A 64% probability of succeeding in the role

Exact matched and missing mandatory skills are shown separately. Since semantic
similarity carries a 20% weight, a semantic score of 64% contributes
`64 × 0.20 = 12.8` points to the final score.

### Mandatory-skill score

This is exact normalized coverage. If a job requires four mandatory skills and the
candidate has three, the mandatory score is 75%. The result card lists both the matched
and missing skills.

### Experience score

- A candidate inside the job's range receives 100%.
- A candidate below the minimum receives a proportional score.
- A candidate above the maximum receives a proportional penalty.
- A job with no experience range receives 100% for this component.

Scores are ranking aids for recruiters, not automated hiring decisions.

## Result cards and CSV export

Each result card displays:

- Rank, job title, reference number, creation date, status, Geo, and experience range
- Final match score and progress indicator
- Open positions, designation, and business unit
- Mandatory-skill score, experience score, and semantic similarity
- Matched and missing mandatory skills
- Referral availability

The downloadable CSV adds candidate details and one row per returned job. It includes
the recruiter-facing job fields, individual scores, and mandatory-skill gaps. The file
name is generated safely from the candidate's name.

## Performance behavior

- Ollama chat and embedding models are kept warm for 30 minutes.
- ATS records that have not changed are not embedded again.
- The current Streamlit session caches up to five prepared resumes.
- A prepared resume contains the validated candidate profile and candidate embedding.
- Reanalyzing the same PDF with different filters reuses extraction and embedding work.
- The cache key includes the PDF content and configured model names.
- **Start over** clears the session's prepared-resume cache.
- The UI reports time spent on PDF extraction, candidate extraction, candidate
  embedding, vector search, and deterministic matching.

The first model request after Ollama has been idle is normally slower because the model
must be loaded. Later requests are usually faster while the model remains warm.

## Storage and practical data capacity

TalentFit does not impose a fixed ATS row limit in application code. Practical capacity
depends on the Streamlit upload limit, available memory, Ollama embedding speed, and
local disk space. The repository includes 100-job and 500-job files for functional and
performance testing.

ChromaDB data is written to `CHROMA_PATH` (`data/chroma` by default). It remains available
after Streamlit or the computer restarts. The directory is intentionally ignored by Git
because it is generated local data and may contain real ATS content.

For larger datasets, upload a complete snapshot, allow initial embedding to finish, and
use later incremental snapshots so unchanged jobs can be skipped.

## Project structure

```text
TalentFit/
├── streamlit_app.py                 # Streamlit entry point
├── pyproject.toml                   # Package metadata and dependencies
├── requirements.txt                 # Editable install with development dependencies
├── .env.example                     # Safe local configuration template
├── README.md                        # Main application knowledge base
├── src/app/
│   ├── ats_service.py               # ATS reading, aliases, and value normalization
│   ├── candidate_extractor.py       # Qwen extraction and experience reconciliation
│   ├── config.py                    # Environment-based configuration
│   ├── document_builder.py          # Stable job and candidate search documents
│   ├── embedding_service.py         # Ollama embedding requests
│   ├── indexing_service.py          # Snapshot insert/update/skip/delete orchestration
│   ├── matcher.py                   # Explainable deterministic scoring
│   ├── resume_matching_service.py   # Resume preparation, search, ranking, and timings
│   ├── resume_service.py            # PDF text extraction
│   ├── search_service.py            # Candidate retrieval helper
│   ├── skill_normalizer.py          # Skill cleanup and aliases
│   ├── ui.py                        # Streamlit pages, state, cards, and CSV export
│   ├── vector_store.py              # ChromaDB persistence, migration, and filters
│   └── models/
│       ├── candidate_profile.py     # CandidateProfile model
│       ├── job.py                   # Job model
│       └── match_result.py          # MatchResult model
├── resources/                       # Safe sample ATS and resume files
├── docs/use-cases/                  # Focused use-case walkthroughs
├── tests/                           # Automated unit and integration-style tests
└── data/chroma/                     # Generated local index; ignored by Git
```

## Requirements

- Python 3.11 or newer
- Ollama installed and reachable from the application
- Local disk space for Ollama models and ChromaDB

The project uses Streamlit, Pydantic, Pandas, PyMuPDF, Ollama, ChromaDB,
Beautiful Soup, OpenPyXL, and xlrd. Pytest is included as a development dependency.

## Installation

Clone the repository and enter it:

```bash
git clone <your-repository-url>
cd TalentFit
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the local environment file:

```bash
cp .env.example .env
```

Install the default Ollama models:

```bash
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:0.6b
```

Confirm that they are present:

```bash
ollama list
```

The expected model names include:

```text
qwen3:1.7b
qwen3-embedding:0.6b
```

## Configuration

Configuration is loaded from the project-root `.env` file:

| Variable | Default in `.env.example` | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Resume candidate-extraction model |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Job and candidate embedding model |
| `CHROMA_PATH` | `data/chroma` | Persistent local vector database directory |
| `COLLECTION_NAME` | `talentfit_jobs` | ChromaDB job collection name |
| `TOP_K_RETRIEVAL` | `10` | Eligible semantic jobs retrieved before scoring |
| `TOP_K_RESULTS` | `3` | Final ranked jobs shown and exported |

`TOP_K_RETRIEVAL` and `TOP_K_RESULTS` must be positive integers. If a model name is
changed, pull that model in Ollama before starting the application.

## Run or rerun the application

From the repository root:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

Open <http://localhost:8501> if the browser does not open automatically. Streamlit
normally reloads saved code changes automatically. To perform a clean rerun, stop the
server with `Ctrl+C` and run the same command again.

If port 8501 is already in use:

```bash
streamlit run streamlit_app.py --server.port 8502
```

## Five-minute end-to-end demo

1. Start Ollama and the Streamlit application.
2. Open **ATS Upload**.
3. Upload `resources/sample_ats_4_500_jobs.xlsx`.
4. Review the preview and select **Confirm and index**.
5. Open **Job Dashboard** and test the open/referral filters.
6. Open **Resume Match**.
7. Upload any sample resume PDF from `resources/`.
8. Keep **Open jobs only** enabled and select **Analyze resume**.
9. Review the extracted profile, scores, and skill gaps.
10. Download the CSV report.

For a focused synchronization demo, first upload `sample_ats_1.xlsx`, then upload
`sample_ats_2_incremental.xlsx`. The second snapshot demonstrates insert, update, skip,
and delete behavior.

## Sample resources

| File | Purpose |
|---|---|
| `sample_ats_1.xlsx` | Four-job baseline snapshot with extra ignored columns |
| `sample_ats_2_incremental.xlsx` | Snapshot demonstrating insert, update, skip, and deletion |
| `sample_ats_3_100_jobs.xlsx` | Medium ATS upload and performance test |
| `sample_ats_4_500_jobs.xlsx` | Larger independent dashboard and matching demo |
| `sample_jobs.csv` | Compact legacy CSV fixture used by automated tests |
| `sample_resume.pdf` | Java backend candidate |
| `sample_resume_data_ai_engineer.pdf` | Data and AI engineering candidate |
| `sample_resume_cloud_sre.pdf` | Cloud platform and SRE candidate |

See [resources/README.md](resources/README.md) for the recommended sample workflow.

## Tests

Run the complete suite from the repository root:

```bash
source venv/bin/activate
python -m pytest -q
```

The tests cover ATS parsing, candidate extraction, document building, embeddings,
incremental indexing, matching, PDF extraction, search, skill normalization, Streamlit
helpers, and ChromaDB behavior. GitHub Actions runs the suite on pushes and pull
requests.

## Security and Git safety

- `.env` is local-only and ignored by Git.
- `.env.example` contains no credentials and is safe to commit.
- `.streamlit/secrets.toml` is ignored and must never be force-added.
- `data/chroma` is ignored because it is generated and may contain ATS job data.
- Virtual environments, temporary files, generated output, local agent metadata, IDE
  settings, and OS files are ignored.
- Sample resumes contain fictional identities and contact information.
- Uploaded files are processed through temporary directories and are not copied into the
  repository.
- Before pushing, inspect `git status`, `git diff --cached`, and ignored files.

Do not commit real ATS exports, candidate resumes, access tokens, passwords, or a local
`.env` file.

## Troubleshooting

### Ollama connection error

Confirm that Ollama is running, `OLLAMA_HOST` is correct, and the configured models
appear in `ollama list`.

### Model not found

Pull the exact model named in `.env`, then restart or rerun the analysis:

```bash
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:0.6b
```

### ATS file is missing required columns

Confirm that the file contains a recognized Reference Number header and Job Title
header. Header casing does not matter, but unsupported names are ignored.

### Jobs unexpectedly disappeared after upload

Each ATS upload is a full snapshot. Jobs absent from the newest file are intentionally
deleted. Re-upload the correct complete snapshot to restore them.

### No jobs appear after filtering

Disable one or both filters, or upload a snapshot containing jobs with status `open`
and/or referral enabled.

### Resume has no extractable text

The PDF is probably scanned or image-only. Convert it to a searchable PDF with OCR
before uploading.

### Resume experience looks incorrect

Check whether the resume contains an explicit total and clear employment start/end
dates. TalentFit excludes education periods and merges overlapping employment, but
ambiguous or missing resume evidence can still affect extraction.

### First resume analysis is slow

The model may be loading into memory. Repeat analysis is faster while Ollama remains
warm, and the same resume can reuse its session-cached profile and embedding.

### Port 8501 is already in use

Start Streamlit on another port:

```bash
streamlit run streamlit_app.py --server.port 8502
```

## Current limitations

- The UI accepts resumes only as PDFs with an extractable text layer.
- Candidate extraction quality depends on the clarity of resume content and the chosen
  Ollama model.
- The current ATS contract contains mandatory skills but no optional-skills column.
- The application is designed for a single trusted local user or team.
- Authentication, authorization, audit history, and multi-tenant isolation are not
  implemented.
- ChromaDB is local; a hosted deployment needs a persistent writable volume and an
  explicit data-protection plan.
- Match scores assist human review and must not be treated as autonomous hiring
  decisions.

## Detailed use-case guides

- [ATS upload and indexing](docs/use-cases/ats-upload.md)
- [Incremental ATS synchronization](docs/use-cases/incremental-sync.md)
- [Indexed job dashboard](docs/use-cases/job-dashboard.md)
- [Resume analysis and job matching](docs/use-cases/resume-matching.md)
