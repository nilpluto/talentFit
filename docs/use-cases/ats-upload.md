# Use case: ATS upload and indexing

## What this does

This flow takes a CSV or Excel ATS export and makes its jobs searchable in TalentFit.

## User steps

1. Open **ATS Upload**.
2. Select a CSV, XLS, or XLSX file.
3. Review the first ten normalized rows.
4. Select **Confirm and index**.
5. Wait for the inserted, updated, unchanged, and removed counts.

## Simple flow

```text
ATS file
  -> keep supported columns
  -> clean values and skills
  -> create Job objects
  -> create search text
  -> create embeddings with Ollama
  -> store jobs in ChromaDB
```

## Important behavior

- `Reference Number` and `Job Title` are required.
- The other supported ATS fields are optional.
- Extra columns are ignored.
- Duplicate reference numbers stop the upload with a clear error.
- HTML and unrelated descriptions are not indexed.
- The upload becomes the current ATS snapshot.

## Files involved

| File | Responsibility |
|---|---|
| `src/app/ui.py` | Upload control, preview, confirmation, and summary |
| `src/app/ats_service.py` | Reads the file and normalizes headers and values |
| `src/app/models/job.py` | Validates every normalized job |
| `src/app/skill_normalizer.py` | Normalizes skill names such as `PowerBi` |
| `src/app/document_builder.py` | Builds the text used for semantic search |
| `src/app/embedding_service.py` | Requests embeddings from Ollama |
| `src/app/indexing_service.py` | Coordinates inserts, updates, skips, and deletes |
| `src/app/vector_store.py` | Writes jobs and metadata to ChromaDB |

## Try it

Upload `resources/sample_ats_1.xlsx`. Four jobs should be indexed.
