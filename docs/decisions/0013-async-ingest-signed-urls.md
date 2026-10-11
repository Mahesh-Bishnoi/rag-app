# ADR-0013: Async ingest via signed URLs (supersedes ADR-0009 sync upload)

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-0009 wires `POST /v1/kb/upload` synchronously: validate → load →
  split → embed → blob put → persist. Any embedding-provider outage turns
  uploads into `500`, and the API proxies every byte.
- Decision: uploads return `200` on a good upload; embedding runs
  async in a background worker; per-file stage is retrievable; uploads use
  signed (direct-to-blob) URLs per object-store best practice.
- Scope answers (user, 2026-10-11): (1) remove the sync multipart
  endpoint, no dual paths; (2) fine-grained stages; (3) typed failure
  codes, still generic (no internals leak).
- Schema note: still in development, no prod data. No migration chain —
  the baseline (migration `0001`) is rewritten to include the new columns
  and the old baseline is dropped. Existing dev volumes must be recreated
  (`compose down -v && compose up -d && alembic upgrade head`); `stamp`
  history from ADR-0006 no longer applies.

## Decision
1. Sync `POST /v1/kb/upload` is removed (route, `_upload_file` multipart
   path, `UploadResponse`). One ingest path only.
2. New flow:
   - `POST /v1/kb/uploads/initiate` `{filename, content_type,
     size_bytes}` → validates MIME allowlist, 20MB cap, sanitized name;
     inserts `documents` row with `status='initiated'`; returns
     `{document_id, upload_url, expires_in}`. The URL is a short-lived
     Azurite SAS (write/create only, ~15 min); the in-memory fake mints
     equivalent tokens.
   - Client `PUT`s bytes directly to the URL; the API never proxies file
     bytes.
   - `POST /v1/kb/uploads/{id}/complete` → server HEADs the blob (exists,
     size ≤ cap, type match) → `status='uploaded'`; mismatch/missing →
     typed failure, no silent accept.
   - Worker (new `src/rag_lab/worker/` entrypoint, `worker` service in
     `compose.yaml`, poll interval in `Settings`) claims `uploaded` rows
     (`SELECT ... FOR UPDATE SKIP LOCKED`) and advances
     `uploaded → chunking → embedding → finished`. Chunk rows are written
     in the same transaction as the `finished` flip; each stage is its own
     commit so a crash retries idempotently (re-download, re-derive).
   - `GET /v1/kb/documents/{id}` → `{status, chunk_count, error_code}`
     at any time.
3. `documents` carries `status TEXT NOT NULL` (check constraint over the
   stage set below) + `error_code TEXT NULL` (+ partial index on
   `status = 'uploaded'` for the poll). Baseline migration rewritten, not
   appended (see Context).
4. Stage set: `initiated`, `uploaded`, `chunking`, `embedding`,
   `finished`, `failed` (`failed` terminal, always paired with an
   `error_code`).
5. Typed codes (generic, client-renderable): `upload_incomplete`,
   `blob_missing`, `size_exceeded`, `type_mismatch`, `decode_failed`,
   `empty_content`, `embed_failed`, `persist_failed`. Provider/DB causes
   are logged server-side only, never in the code.

## Alternatives Considered

### Keep the sync endpoint as a dev path
- Pros: Zero test churn; synchronous path stays for debugging
- Cons: Two ingest paths (sync/async) sharing loaders but diverging on
  errors, status, and blob flow — the exact drift the port discipline
  exists to prevent
- Rejected: user decision; one path

### Coarse status (pending/ready/failed)
- Pros: Smaller state machine
- Cons: Loses the "where did it stall" observability the user explicitly
  wants (uploaded vs chunking vs embedding)
- Rejected: fine-grained costs one column either way

### Free-text error messages
- Pros: Easiest to write at each failure site
- Cons: Leaks internals across the API; clients string-match
- Rejected: closed code set above

### OS cron driving the worker
- Pros: Literal match to "cron jobs"
- Cons: A cron daemon inside/beside containers to manage, log, and
  observe; at one-repo scale a poll loop in the worker process has
  identical semantics with one fewer moving part
- Rejected: worker poll loop with `INGEST_POLL_SECONDS` in `Settings`

## Consequences
- Upload availability no longer depends on the embedding provider;
  provider outage surfaces as stalled/`failed` documents, never a failed
  upload.
- `StoragePort` grows `issue_upload_url` + `download` (Azurite SAS +
  `get_blob`; fake equivalents). `put` stays for tests/fakes.
- Baseline rewrite: `alembic` history restarts at the new `0001`; any
  volume stamped under the old baseline must be recreated (command in
  Context). README quick-start gains the recreate note.
- Hermetic tests move to initiate → fake-put → complete → drain-worker;
  live tier (`RAGLAB_LIVE_DB=1`) proves real SAS PUT to Azurite plus
  worker drain with row cleanup.
- A cosine/HNSW vector index stays deferred to the retrieval slice, when
  the query pattern is known.
