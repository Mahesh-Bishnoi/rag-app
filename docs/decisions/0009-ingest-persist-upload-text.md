# ADR-0009: Upload wires ingest-persist (text) behind a DocumentStore port

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-0005 left `POST /v1/kb/upload` as blob-only and `POST /v1/chat/completions`
  as an honest `501`. The next slice (plan track 1) wires upload through
  loader dispatch + blob put + transactional `DocumentRow`/`ChunkRow` persist.
- Constraints: table rows never leave endpoints / DTOs never persisted
  (ADR-0008 discipline); hermetic tests by default, live behind
  `RAGLAB_LIVE_DB=1`; ruff complexity ≤10; layers
  `api → adapters/ingest → core` kept.

## Decision
1. New `adapters/documents.py` owns the save step behind `DocumentStorePort`
   (`save(document: Document, chunks: list[Chunk]) -> str` document id).
   `PostgresDocumentStore` (SQLModel `Session`, one transaction, ids from
   `gen_random_uuid()`) is prod; `InMemoryDocumentStore` is the hermetic fake.
   Two adapters from day one justifies the seam.
2. `POST /v1/kb/upload` orchestrates visibly: validate → dispatch
   `LoaderPort.can_handle` → `load()` → put blob → `save()` → respond.
   Load runs before blob put so undecodable bytes never orphan a blob.
   `UploadResponse` gains `document_id` + `chunk_count` (additive, old fields
   unchanged).
3. Honest failures, generic details: no loader (PDF today) → `415`;
   non-UTF-8/empty decode → `422`; persist failure → `500 ingest failed`
   (cause logged server-side only).
4. Complexity budget: endpoint body lives in module-level `_upload_file`
   with pure `_select_loader` / `_load_chunks` helpers; the route wrapper only
   injects `app.state` collaborators.

## Alternatives Considered

### Persist inline in the endpoint (no port)
- Pros: One less abstraction
- Cons: Hermetic upload tests would need Postgres; no seam for the Phase 3
  retrieval writer to reuse
- Rejected: Breaks the hermetic-by-default bar

### SQLite-backed tests instead of a fake
- Pros: Real SQL in hermetic tests
- Cons: `Vector(768)`, `TSVECTOR`, `PG_UUID` are Postgres-only; the test
  schema would diverge from migration 0001
- Rejected: Fake + live round-trip proves more (contract + real PG types)

### Blob put before load
- Pros: blob_url available earlier
- Cons: undecodable uploads orphan blobs with no delete port (YAGNI today)
- Rejected: Load-first ordering avoids the orphan class entirely

## Consequences
- PDF uploads return `415` until the PDF-loader slice lands; no code change
  needed there — just register the loader.
- `create_app` takes an optional `doc_store`; tests inject the fake, prod
  builds `PostgresDocumentStore` at the composition root.
- DB failure after blob put can orphan a blob (accepted lab limit; a delete
  on the `StoragePort` is the later fix if it hurts).
- Live proof: `PostgresDocumentStore` save round-trip (write → read →
  delete) plus the existing migration upgrade/downgrade round-trip.
