# ADR-0014: Retrieve → rerank → generate over ingested chunks

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-0013 landed async ingest: `documents`/`chunks` tables carry `Vector(768)`
  + `TSVECTOR` with a GIN index, but no vector index and no query path.
  `POST /v1/chat/completions` returns an honest `501` (`src/rag_lab/api/main.py:202`).
- Intent + glossary promise hybrid BM25+vector with RRF, a rerank pass, cited
  generation, and per-question traces.
- Scope answers (user, 2026-10-11): (1) embedding dim is read from the model
  at request time with `768` as DDL/stub fallback; (2) rerank is heuristic
  for v1; (3) chat keeps the strict OpenAI shape plus an extra `citations` field.

## Decision
1. Migration `0002` adds `chunks_embedding_hnsw_idx`
   (`USING hnsw (embedding vector_cosine_ops)`); `tables.py` metadata declares
   the same index. No column change.
2. New `adapters/retrieval.py` with `RetrievalPort.search(query_embedding,
   query_text, limit)`: Postgres runs vector-cosine top-K plus
   `ts_rank`/`plainto_tsquery` top-K and fuses with RRF (`k=60`);
   `InMemoryRetrieval` (cosine + small BM25 in pure Python) is the hermetic fake.
3. Dim discipline: the query path uses `len(query_vector)` as read from the
   model; `Settings.embedding_dimensions` (default `768`) is the DDL truth and
   `PostgresRetrieval` rejects mismatches loudly (`ValueError` → `500` + log),
   never silently.
4. Rerank is a pure function (`adapters/rerank.py`): token overlap +
   exact-phrase bonus over RRF order, deterministic tiebreaks, truncate to
   `answer_context_chunks`. No port: one algorithm does not earn a seam;
   the port arrives with the second (cross-encoder) algorithm.
5. Generation reuses `LlmPort.chat` (+ `StubLlm` fake); the prompt numbers
   contexts `[1..n]` and instructs the model to cite them.
6. `POST /v1/chat/completions` wires embed → retrieve → rerank → generate and
   returns `{id, object: "chat.completion", created, model, choices, citations}`.
   Empty KB is `200` with `citations: []`, not an error. Failures are `500`
   with server-side logs; provider text never leaks.
7. Traces are structured logs (`event="qa_answered"` with `trace_id`, query,
   retrieved ids/scores, `rerank_order`, context ids) — no trace table in v1.
8. Knobs in `Settings`: `retrieval_candidates=40`, `answer_context_chunks=5`.

## Alternatives Considered

### pg_trgm similarity instead of tsvector rank
- Pros: Already-enabled extension; typo-tolerant
- Cons: `text_search` + GIN ships in the baseline; trigram is a second
  keyword path with its own index to keep
- Rejected: rank over the existing tsvector column

### Cross-encoder rerank now
- Pros: Best relevance per the glossary's rerank definition
- Cons: New model download + serving path on a 13.5GB laptop before the
  pipeline works end to end
- Rejected for v1: heuristic proves the seam; cross-encoder is the
  documented upgrade

### Separate KB answer endpoint instead of OpenAI shape
- Pros: Freedom to shape `{answer, citations, trace_id}` exactly
- Cons: Splits clients between two chat APIs; OpenAI-compatible was the
  skeleton's stated surface
- Rejected: strict shape + extra `citations` (user decision)

### Persist traces to a table
- Pros: Queryable history for eval
- Cons: Schema + retention + API surface before any trace exists to inspect
- Rejected for v1: logs carry the fields; RAGAS eval reads logs/DTOs

## Consequences
- Chat availability depends on the embedding provider at query time (ingest
  stays independent per ADR-0013); embed/retrieve/LLM failures are `500`s
  with typed logs.
- `test_migrations` expects head `0002`; volumes upgrade with
  `alembic upgrade head` (index builds instantly on small tables).
- Hermetic tests cover fusion/rerank/chat via fakes; live tier proves the
  real HNSW + `ts_rank` SQL with row cleanup.
- A model with dim != 768 needs a migration changing `vector(n)` plus a
  `Settings.embedding_dimensions` bump — deliberate, recorded, not silent.
