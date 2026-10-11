# Architecture — rag-lab (human map)

Read this before opening `src/`. Terms follow `docs/glossary.md`; decisions live in `docs/decisions/`.

## Mental model

A local RAG lab: drag-drop files → async ingest → hybrid search → cited answer. The API never
proxies file bytes; the browser PUTs directly to a signed blob URL.

```
upload initiate → PUT bytes → complete → worker drain → chat → trace
```

* `POST /v1/kb/uploads/initiate` mints a 15-min write-only URL + `initiated` row.
* Client `PUT`s bytes to that URL (with `Content-Type` + `x-ms-blob-type: BlockBlob`).
* `POST /v1/kb/uploads/{id}/complete` HEADs the blob, flips `initiated → uploaded`
  (or `failed` with a typed `error_code` — the body carries the error, not just the status).
* Worker `drain`s `uploaded → chunking → embedding → finished` (`graph/ingest.py` owns the order).
* `POST /v1/chat/completions` runs `graph/qa.py`: embed → retrieve → rerank → generate,
  returns OpenAI shape + top-level `citations`, logs `event="qa_answered"` with `trace_id`.
* `GET /v1/traces/{trace_id}` replays `{query, retrieved, rerank_order, prompt, answer, citations}`.

## Where code lives

* `api/` — HTTP only. `schemas.py` (wire DTOs), `deps.py` (resolve ports from `app.state`
  or build prod defaults), `main.py` (route handlers + `create_app`). No SQL, no blob IO here.
* `graph/` — pipeline order. `qa.py` (4 nodes), `ingest.py` (5 nodes + `run_ingest_one`).
  Nodes are thin; ports do the work.
* `adapters/` — ports + two implementations each: Postgres/Azurite for prod,
  in-memory fakes for hermetic tests. `tables.py` mirrors migrations exactly (never import
  Pydantic DTOs there, never return rows from endpoints).
* `ingest/` — `loaders.py` (`TextLoader`, `PdfLoader`), `chunking.py` (heading-aware split).
* `core/` — pure domain: `models.py` (DTOs), `config.py` (`Settings`, loaded once at startup),
  `text.py` + `scoring.py` (tokenize, cosine, BM25, RRF — no IO).
* `worker/` — `pipeline.drain/process_one` loop over `run_ingest_one`; `run.py` entrypoint.
* `eval/` — heuristic RAGAS-style harness over `eval/golden.jsonl`.

Layer rule (`.importlinter`): `api → eval/worker/graph → adapters/ingest → core`.
Dependencies point inward; `core` imports nothing from the app.

## Ports and fakes

| Port | Prod | Fake (tests) | Seeded via |
|---|---|---|---|
| `DocumentStorePort` | `PostgresDocumentStore` | `InMemoryDocumentStore` | `initiate()` |
| `RetrievalPort` | `PostgresRetrieval` | `InMemoryRetrieval` | `add()` |
| `StoragePort` | `AzuriteBlobStorage` | `InMemoryStorage` | `put()` |
| `EmbeddingPort` | `OpenAICompatibleEmbedding` | `StubEmbedding` | constructor |
| `LlmPort` | `OpenAICompatibleLlm` | `StubLlm` | constructor |
| `TraceStorePort` | `PostgresTraceStore` | `InMemoryTraceStore` | `save()` |

One adapter is hypothetical; two is a real seam. Don't add a new port for a single algorithm
(rerank stays a plain function until the cross-encoder arrives — ADR-0014).

## Debugging entry points

* Upload stuck `initiated`? `complete` HEAD failed or blob missing → check `error_code`
  (`upload_incomplete`, `size_exceeded`, `type_mismatch`, `empty_content`).
* Ingest stuck `uploaded`? Worker not running (`python -m rag_lab.worker.run`) or
  `claim_next` lost the race (by design: `SKIP LOCKED`).
* Chat has no citations? Empty KB returns `200` + `citations: []` — seed or ingest first.
* Wrong chunks? Check `GET /v1/traces/{trace_id}` (`retrieved` scores + `rerank_order`)
  and the `qa_answered` log line before touching the prompt.

Decisions: `docs/decisions/` (0001–0018, never rewrite closed ones — add a new numbered ADR
that supersedes). Start with ADR-0013 (async ingest), ADR-0014 (retrieve→rerank→generate),
ADR-0015 (traces + eval).
