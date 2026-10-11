# ADR-0012: Embed-on-ingest behind an EmbeddingPort (split LLM adapter)

## Status
Accepted

## Date
2026-10-11

## Context
- Upload persisted chunks with `embedding=NULL` (column exists since
  migration 0001, writer landed as "Phase 2" in ADR-0006). Retrieval needs
  vectors at write time.
- `LlmPort` bundled `chat + embed` behind one model, contradicting
  `Settings` (separate `openai_*` vs `embedding_*`) — every embed call
  would have used the chat model.
- Constraints: hermetic tests by default; rows never leave endpoints / DTOs
  never persisted directly (ADR-0008 discipline); upload stays honest
  (`415/422/500` per ADR-0005/0009); ruff complexity ≤10.

## Decision
1. Split the transport: `LlmPort` keeps `chat` only
   (`adapters/llm_openai.py`); new `EmbeddingPort` (`adapters/embeddings.py`)
   with `OpenAICompatibleEmbedding(base_url, api_key, model)` for the
   `/v1/embeddings` call plus a deterministic `StubEmbedding(dim=768)`
   fake for hermetic tests. Two adapters justify the seam.
2. `Chunk` DTO gains `embedding: list[float] | None`; both stores copy the
   field into `ChunkRow.embedding` (same field-copy pattern as
   content/title — the DTO object itself is never persisted).
3. `POST /v1/kb/upload` orchestrates visibly: validate → load → split →
   embed → blob put → persist. Embed runs before blob put so a dead
   provider never orphans a blob. Length mismatch or provider error maps
   to generic `500 ingest failed` (cause logged server-side only).
4. `create_app` takes optional `embedder` (`None` = skip, hermetic-safe
   default); the module composition root builds the real embedder from
   `embedding_*` settings so `fastapi dev`/`run` writes vectors.

## Alternatives Considered

### Embed inside DocumentStore.save
- Pros: One call site; persistence owns the vector column
- Cons: Store gains network IO + provider config; hermetic upload tests
  need a fake inside the store; hides a side effect inside `save`
- Rejected: orchestrator lists each IO (`split → embed → put → save`);
  the store stays pure persistence

### Parallel embeddings list alongside chunks
- Pros: `Chunk` DTO stays text-only
- Cons: Two lists must stay aligned across split/embed/save; every
  caller re-implements the zip
- Rejected: vector rides on the chunk it belongs to; alignment by
  construction

### Embed after blob put (background worker)
- Pros: Upload stays fast when the provider is slow
- Cons: New queue/callback machinery; reads race unwritten vectors;
  contradicts the visible-pipeline success criterion
- Rejected for v1: synchronous first; a worker is the later fix if
  provider latency hurts

## Consequences
- Uploads now require a reachable embedding provider in dev/prod
  (default Ollama `nomic-embed-text`); provider outage → `500`.
- Dim assumption `768` matches `Vector(768)` and the configured model; a
  wrong-dim provider surfaces as a DB error → `500`, logged.
- No migration (column + GIN index unchanged); a cosine/HNSW vector index
  is deferred to the retrieval slice when the query pattern is known.
- Live proof: `PostgresDocumentStore` save round-trip with 768-dim vectors
  plus the existing migration round-trip (`RAGLAB_LIVE_DB=1`).
