# Use case: resume analysis and job matching

## What this does

This flow turns a text-based PDF resume into a candidate profile, finds relevant indexed
India jobs, and returns up to three explainable matches. Jobs outside India and jobs
without at least one matched mandatory skill are excluded.

## User steps

1. Index an ATS file first.
2. Open **Resume Match**.
3. Upload a text-based PDF resume.
4. Choose whether to search only open jobs.
5. Select **Analyze resume**.
6. Review the candidate profile and up to three results.
7. Download the CSV report if needed.

## Simple flow

```text
Resume PDF
  -> extract selectable text
  -> extract CandidateProfile with Qwen
  -> verify experience using employment dates
  -> create candidate embedding
  -> restrict Geo to India and apply the open-job filter
  -> search ChromaDB
  -> score mandatory skills, experience, and semantic similarity
  -> exclude jobs with no mandatory-skill match
  -> show up to three results and skill gaps
```

## Matching output

Each result shows:

- Final match score
- Mandatory-skill score
- Experience score
- Semantic similarity
- Matched mandatory skills
- Missing mandatory skills
- Recruiter-facing ATS job details

## What “Semantic 64%” means

The semantic percentage measures how similar the overall meaning of the candidate
profile is to the indexed job document. The embedding model considers related context,
not only exact word matches.

Candidate context includes:

- Roles
- Skills
- Verified experience
- Location
- Profile summary

Indexed job context includes:

- Job title and designation
- Mandatory skills
- Experience range
- Geo and business unit
- Job status

Therefore, **Semantic 64%** means moderate contextual similarity. It does not mean the
candidate matched 64% of the mandatory skills, and it is not a 64% probability of being
hired or succeeding in the role. The result card shows exact matched and missing
mandatory skills separately.

Semantic matching can recognize related wording. For example, a candidate described as
a “Java microservices lead” may be relevant to a “Backend Engineering Lead” job even
when the titles are not identical.

## How the final score is calculated

| Component | Weight | Meaning |
|---|---:|---|
| Mandatory skills | 50% | Canonical, variant-aware mandatory-technology coverage |
| Optional skills | 15% | Optional-skill coverage; full component when the job specifies none |
| Experience | 15% | Fit against the job's minimum and maximum experience |
| Semantic similarity | 20% | Overall contextual similarity |

Example: a semantic score of 64% contributes `64 × 0.20 = 12.8` points to the final
score. The other three components provide the remaining points. This design keeps exact
mandatory skills more important than semantic similarity while still recognizing
related roles and terminology.

Mandatory matching uses controlled aliases for different names of the same technology,
including `PowerBi`/`Power BI Desktop`, `Service Now`/`ServiceNow`, `.NET`/`Dot Net`,
`React.js`/`React JS`, and `ADF`/`Azure Data Factory`. It deliberately avoids general
fuzzy matching, so Java does not match JavaScript and React does not match React Native.

## Performance behavior

TalentFit keeps Ollama models warm for 30 minutes. It also caches the prepared candidate
and embedding for up to five resumes inside the current Streamlit session. Changing a
filter and analyzing the same resume again skips PDF parsing, Qwen extraction, and
candidate embedding.

Selecting **Start over** clears the session resume cache.

## Files involved

| File | Responsibility |
|---|---|
| `src/app/resume_service.py` | Extracts text from the PDF |
| `src/app/candidate_extractor.py` | Produces the structured profile and verifies experience |
| `src/app/models/candidate_profile.py` | Validates the candidate data |
| `src/app/embedding_service.py` | Creates the candidate embedding |
| `src/app/resume_matching_service.py` | Coordinates preparation, caching, search, and matching |
| `src/app/vector_store.py` | Applies job filters and performs semantic retrieval |
| `src/app/matcher.py` | Calculates explainable deterministic scores |
| `src/app/ui.py` | Upload, filters, result cards, and CSV download |

## Supported and unsupported resumes

- Different text layouts, headings, section orders, and writing styles are supported.
- Multi-page text PDFs are supported.
- Image-only or scanned PDFs are not yet supported because they require OCR.

## Try it

- `resources/sample_resume.pdf` for a Java backend profile
- `resources/sample_resume_data_ai_engineer.pdf` for a data and AI profile
- `resources/sample_resume_cloud_sre.pdf` for a platform and SRE profile
