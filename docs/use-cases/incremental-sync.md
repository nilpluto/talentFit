# Use case: incremental ATS synchronization

## What this does

TalentFit keeps the ChromaDB index aligned with the latest complete ATS export.

## Decision for each reference number

```text
Not already indexed
  -> insert

Already indexed and content changed
  -> update

Already indexed and content unchanged
  -> skip

Indexed before but absent from the latest file
  -> delete
```

TalentFit creates a content hash from the normalized job and its searchable document.
This avoids generating another embedding for an unchanged job.

Only India jobs participate in synchronization. Non-India and missing-Geo rows are not
embedded or stored. Any older non-India records are deleted on the next ATS upload.

## User steps

1. Upload `resources/sample_ats_1.xlsx`.
2. Confirm and index it.
3. Upload `resources/sample_ats_2_incremental.xlsx`.
4. Confirm and index it.
5. Review the synchronization summary.

The second file demonstrates one insert, one update, two unchanged jobs, and one
deletion.

## Files involved

| File | Responsibility |
|---|---|
| `src/app/indexing_service.py` | Computes hashes and decides insert/update/skip/delete |
| `src/app/vector_store.py` | Reads hashes, upserts jobs, and deletes missing jobs |
| `src/app/document_builder.py` | Produces stable searchable text for hashing |
| `src/app/ui.py` | Displays the synchronization result |

## Important warning

An upload is a full snapshot, not a partial patch. Uploading a small file after a large
file removes every previously indexed job that is not present in the small file.
