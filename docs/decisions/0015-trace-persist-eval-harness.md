# ADR-0015: Persist QA traces + heuristic eval harness v1

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-0014 ships retrieve → rerank → generate with per-question traces as
  structured logs only (`event="qa_answered"` in `src/rag_lab/api/main.py`).
  Logs prove the fields exist but are awkward to query per question.
- Intent + glossary promise RAGAS-style eval (scored set over own docs
  measuring retrieval + groundedness) and observability (query, retrieved
  chunks + scores, rerank order, final prompt per question).
- Scope answers (user, 2026-10-11): (1) trace-write failure stays `200`
  with a server-side log (fail-open — the answer succeeded); (2) the trace
  endpoint returns the full record (query, retrieved + scores, rerank
  order, prompt, answer, citations); (3) golden set starts at 12 items
  and grows.
- Hermetic coverage sits at 79.45% against the 80 floor; the new trace
  store + eval code must arrive with tests that lift the gate, not loosen it.

## Decision
1. New `qa_traces` table (migration `0003`): `trace_id TEXT PRIMARY KEY`,
   `query TEXT NOT NULL`, `retrieved JSONB NOT NULL DEFAULT '[]'`
   (list of `{chunk_id, vector_score, bm25_score, fused_rank}`),
   `rerank_order JSONB NOT NULL DEFAULT '[]'`, `prompt TEXT NOT NULL`,
   `answer TEXT NOT NULL`, `citations JSONB NOT NULL DEFAULT '[]'`
   (list of `{chunk_id, source, title, heading_path, page}`),
   `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`.
   `TraceRow` in `adapters/tables.py` mirrors it exactly.
2. New `TraceStorePort` (`adapters/traces.py`) with `save(trace, answer,
   citations)` + `get(trace_id)`: Postgres prod adapter, in-memory fake
   for hermetic tests. Core `QaTrace` DTO is the read shape; table rows
   never leave endpoints (rows map to DTOs at the boundary).
3. Chat wires save as fail-open: embed → retrieve → rerank → generate →
   try-save (exception → `logger.exception`, still `200`) → respond.
   The `qa_answered` log line stays for tailing.
4. `GET /v1/traces/{trace_id}` returns the full `QaTrace` (`404` unknown).
   Chat `id` becomes `chatcmpl-{trace_id}` in full (was truncated to 29
   chars, which made the trace unrecoverable from the id).
5. Eval harness v1 (`src/rag_lab/eval/`): pure heuristic metrics, no new
   model calls —
   - retrieval `hit@k` / `recall@k` / `MRR` over ranked id lists;
   - `keyword_coverage(answer, expected_keywords)` fraction;
   - `groundedness(answer, cited_contents)` token-overlap fraction
     (shared `tokenize` from retrieval);
   - `evaluate_suite` runs `eval/golden.jsonl` (12 seed items over
     intent/glossary/ADRs) against `InMemoryRetrieval` + `rerank_candidates`
     with per-item fixed embeddings (simulated perfect embedder —
     documented limit, not a claim about embedder quality).
   - `python -m rag_lab.eval.run` prints pass/fail per item and exits
     non-zero below thresholds; `tests/test_eval.py` pins the same path.
6. `.importlinter` gains an `rag_lab.eval` layer above `rag_lab.worker`
   (eval consumes adapters + core; API does not consume eval).

## Alternatives Considered

### Keep traces logs-only
- Pros: Zero schema; ADR-0014 already decided it
- Cons: Per-question inspection means grepping logs; no stable read shape
  for the eval harness or debugging UI later
- Rejected: user decision; logs stay as a complement, not the store

### Fail-closed trace writes (500 on save failure)
- Pros: No silent observability loss
- Cons: A working answer becomes an error because bookkeeping failed;
  couples answer availability to the trace table
- Rejected: fail-open per user decision; save failures log loudly

### LLM-judged eval now
- Pros: Closer to RAGAS groundedness/relevance grading
- Cons: New model calls + prompts + flakiness before the harness proves
  the seam; laptop-budget and hermetic-default constraints
- Rejected for v1: heuristics prove the runner + golden-set workflow;
  LLM judge is the documented upgrade

### Cross-encoder rerank now
- Pros: Best relevance
- Cons: Same as above — new serving path before pipeline + eval work end to end
- Rejected for v1 (as in ADR-0014); eval thresholds are set so a future
  rerank swap shows up as metric movement

## Consequences
- Chat availability no longer depends on trace writes; trace loss surfaces
  as error logs, never as failed answers.
- `test_migrations` expects head `0003`; `test_tables` includes `qa_traces`.
- Hermetic tests cover save/get via the fake, fail-open chat, the GET
  endpoint, and the eval suite; live tier proves the Postgres trace
  round-trip with row cleanup.
- Queries + answers are now stored (purpose: per-question debugging +
  eval). No retention/deletion path yet — a future ADR owns TTL/GDPR
  erasure before any multi-user or hosted use.
- Golden set grows by appending JSONL lines + rerunning thresholds;
  thresholds live in `run.py` and change only with a recorded reason.
