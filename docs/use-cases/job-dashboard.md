# Use case: indexed job dashboard

## What this does

The dashboard shows India jobs currently available in the ChromaDB index without
requiring a resume. Jobs from other countries and jobs with no Geo are never indexed.

## User steps

1. Index an ATS file.
2. Open **Job Dashboard**.
3. Leave **Open jobs only** enabled to hide unavailable jobs.
4. Review the metrics and job table.

## Dashboard information

- Total indexed jobs
- Jobs matching the selected filters
- The 9 supported ATS fields for each job

## Simple flow

```text
Geo = India
  -> Open jobs toggle
  -> ChromaDB metadata filter
  -> matching indexed jobs
  -> dashboard metrics and table
```

## Files involved

| File | Responsibility |
|---|---|
| `src/app/ui.py` | Dashboard filters, metrics, and table |
| `src/app/vector_store.py` | Retrieves jobs with native Chroma metadata filters |
| `src/app/models/job.py` | Restores validated Job objects from stored metadata |

## Try it

Index `resources/sample_ats_4_500_jobs.xlsx`, then switch **Open jobs only** on and off.
