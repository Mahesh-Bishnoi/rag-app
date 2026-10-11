# ADR-0005: OpenAI-compatible LLM adapter, Azurite blob store, OpenAI-compatible API surface

## Status
Accepted

## Date
2026-10-11

## Context
- User picks for v1: `uv` toolchain; LLM via OpenAI-compatible adapter (Ollama locally or any provider by base_url swap); `Azurite?`; FastAPI backend exposing OpenAI-compatible API for a future Unsloth-Studio-like frontend (or Unsloth itself); first milestone `infra+skeleton`.
- ADR-002 chose S3-compatible store (MinIO-style). Local box already caches `mcr.microsoft.com/azure-storage/azurite:latest` and `postgres:16`; no MinIO/pgvector image cached. Box is Ryzen 7 5800H + RTX 3050 Mobile + ~13.5GB RAM, podman-first.
- ADR-003 seam rule stands: LangGraph = flow, PydanticAI = typed LLM calls, LangChain = loaders/splitters. The OpenAI-compatible adapter must not become a second agent framework — it is the transport PydanticAI calls through.

## Decision
1. `uv + pyproject` with `src/rag_lab` layout, `requires-python >=3.11`.
2. LLM/embedding access behind one `LlmPort` speaking OpenAI REST (`/v1/chat/completions`, `/v1/embeddings`): `base_url + api_key + model` from settings. Default `http://localhost:11434/v1` (Ollama); swap to any provider without code change.
3. Blob store is **Azurite** (Azure Blob REST) for v1, partially superseding ADR-002's S3 choice: raw uploads (PDF, text) go to Azurite container `kb-raw`; Postgres holds derived chunks only. Storage sits behind a `StoragePort` so an S3 (`boto3`) adapter can be added later without touching ingest.
4. FastAPI exposes OpenAI-compatible surface: `GET /health` (db+blob checks), `GET /v1/models`, `POST /v1/chat/completions` (OpenAI request shape; RAG wiring lands in Phase 3), `POST /v1/kb/upload` (multipart, outside OpenAI spec, documented as extension).
5. Postgres image is `pgvector/pgvector:pg16` (vectors + `tsvector` BM25 path from day one per ADR-002 consequences). `postgres:16` cached image is a fallback only.
6. Milestone order is infra+skeleton first: `podman compose up` + schema + `/health` green before any retrieval logic.

## Alternatives Considered

### MinIO (S3-compatible, ADR-002 default)
- Pros: Keeps ADR-002 verbatim; `boto3` ecosystem; S3 API is the more common RAG tutorial target
- Cons: New image pull; one more API to learn; no cached image on this box
- Rejected for v1: Azurite cache + single Azure SDK wins on ops weight; port keeps S3 return path open

### Filesystem volume as object-store stand-in
- Pros: Zero services, fastest skeleton
- Cons: No blob API to learn, diverges from both ADR-002 and Azurite direction; upload semantics (multipart, ranges) not exercised
- Rejected: Hides the storage seam instead of proving it

### Native provider SDKs (ollama-python, openai-only, azure-openai-only)
- Pros: Richer per-provider features
- Cons: One SDK per provider; every provider swap edits code; violates adapter seam
- Rejected: OpenAI REST is the narrow waist — one client, many backends

### Vertical slice first (in-memory RAG before containers)
- Pros: Fastest learning loop
- Cons: Defers the one-way-door compose/service/schema decisions the user asked to lock first
- Rejected: User explicitly chose infra+skeleton

## Consequences
- `compose.yaml` service names/ports/volume layout are a one-way door: `db` (5432), `azurite` (10000/10001/10002), volumes `pgdata`, `azurite-data` stay stable.
- Schema carries `documents` + `chunks` (text, `vector`, `tsvector`, title, heading path, source, page, mime, offsets) from day one.
- `.env` holds secrets (`DATABASE_URL`, `AZURITE_CONNECTION_STRING`, `OPENAI_API_KEY`); only `.env.example` with placeholders is committed.
- Upload path needs hardening from day one: MIME allowlist, size cap, filename sanitization, generic errors (see hardening skill).
- Phase 3 must wire `POST /v1/chat/completions` through retrieve → rerank → generate with citations + trace; skeleton returns honest `501` until then.
- If S3 parity is ever needed, add `S3StorageAdapter` implementing the same `StoragePort` — no ingest changes.
