# rag-lab

Local RAG lab (infra+skeleton v1). See `docs/intent/rag-lab.md` and `docs/decisions/`.

## Quick start

1. Copy env: `cp .env.example .env`
2. Install: `uv sync --group dev`
3. Infra: `podman compose up -d`
4. Schema (fresh volume): `uv run alembic upgrade head`
5. Run API (dev): `uv run fastapi dev` (reload, entrypoint in `[tool.fastapi]`);
   prod: `uv run fastapi run`
6. Run worker (dev): `uv run python -m rag_lab.worker.run` (prod: `worker`
   service in `compose.yaml`)
7. Checks: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`,
   `uv run mypy .` (strict), `PYTHONPATH=src uv run lint-imports`,
   `uv run deptry src alembic`, `uv audit`
8. Secrets: `gitleaks detect --source . --config .gitleaks.toml` (clean required;
   allowlist in `.gitleaks.toml` covers only well-known non-secrets, with reason)
9. Live tier (needs `podman compose up -d` + `.env`):
   `RAGLAB_LIVE_DB=1 uv run pytest` — migration upgrade/downgrade round-trip
   plus Azurite put/SAS round-trips and worker drain. Focused runs skip the coverage gate with
   `uv run pytest <path> --no-cov`. Stop the worker first
   (`podman stop rag-lab-worker`): it polls the same dev database and will
   claim `uploaded` rows out from under the live document tests.

Migration `0001` was rewritten for async ingest (ADR-0013): any volume
created or stamped under the old baseline must be recreated
(`podman compose down -v && podman compose up -d && uv run alembic upgrade head`).
New migrations: `uv run alembic revision -m "..." --autogenerate` (once ORM
models land and attach to `target_metadata` in `alembic/env.py`).

## API (OpenAI-compatible + KB extension)

- `GET /` → minimal drag-drop UI (upload PDF/text, poll ingest, chat with citations + trace link). API startup writes the browser-upload CORS rule to the local Azurite emulator only (origins `blob_cors_origins`, fail-open), so fresh volumes upload with zero extra steps — see ADR-0018.
- `GET /health` → `{status, db, blob}`
- `GET /v1/models` → configured chat + embedding model ids
- `POST /v1/chat/completions` → hybrid retrieve (vector + BM25, RRF) → heuristic rerank → generate; strict OpenAI shape plus top-level `citations: [{chunk_id, source, title, heading_path, page}]` (empty KB → `200` with `citations: []`); per-question trace in logs (`event="qa_answered"`, `trace_id`, retrieved scores, `rerank_order`)
- `GET /v1/traces/{trace_id}` → full persisted record `{trace_id, query, retrieved, rerank_order, prompt, answer, citations}` (`404` unknown; chat `id` is `chatcmpl-{trace_id}` in full so answers link to traces; trace writes are fail-open — a working answer stays `200` even when the save fails)
- `POST /v1/kb/uploads/initiate` `{filename, content_type, size_bytes}` → `{document_id, upload_url, expires_in}` (15-min write-only SAS; API never proxies bytes)
- client `PUT`s bytes to `upload_url` with matching `Content-Type` and `x-ms-blob-type: BlockBlob` (Azure requires the latter; the UI sends both)
- `POST /v1/kb/uploads/{id}/complete` → server HEADs the blob, flips `initiated → uploaded` (missing → `upload_incomplete`, oversize → `size_exceeded`, type mismatch → `type_mismatch`, empty → `empty_content`)
- worker drains `uploaded → chunking → embedding → finished` (`embed_failed`/`persist_failed`/`decode_failed` on failure)
- `GET /v1/kb/documents/{id}` → `{document_id, status, chunk_count, error_code}` at any time

Default LLM transport points at Ollama (`http://localhost:11434/v1`); swap `OPENAI_BASE_URL` to any OpenAI-compatible provider.

## Layout

Start with `docs/architecture.md` for the request journeys, then:

- `src/rag_lab/core/` — pure domain: `models` (DTOs), `config` (`Settings`), `text` + `scoring` (tokenize, cosine, BM25, RRF — no IO)
- `src/rag_lab/api/` — HTTP only: `schemas` (wire DTOs), `deps` (port resolution), `main` (routes + `create_app`), `ui` + `static/index.html` (`GET /`)
- `src/rag_lab/adapters/` — ports with prod + in-memory fake each: `llm_openai`, `generation`, `embeddings`, `storage`, `documents`, `retrieval`, `rerank_candidates`, `db_ping`, `traces`
- `src/rag_lab/ingest/` — `LoaderPort`, `TextLoader`, `PdfLoader`, `split_chunks` (heading-aware sections)
- `src/rag_lab/graph/` — pipeline order: `qa` (embed → retrieve → rerank → generate), `ingest` (download → load → split → embed → finish)
- `src/rag_lab/worker/` — `pipeline.drain/process_one` over `run_ingest_one`, `run` entrypoint
- `src/rag_lab/eval/` — heuristic RAGAS-style harness over `eval/golden.jsonl`
- `compose.yaml` — `db` (pgvector:pg16, :5432), `azurite` (:10000-10002), `worker` (ingest poll loop)
