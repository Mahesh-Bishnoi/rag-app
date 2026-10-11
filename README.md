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

- `GET /health` → `{status, db, blob}`
- `GET /v1/models` → configured chat + embedding model ids
- `POST /v1/chat/completions` → `501` until Phase 3 wires retrieve → rerank → generate
- `POST /v1/kb/uploads/initiate` `{filename, content_type, size_bytes}` → `{document_id, upload_url, expires_in}` (15-min write-only SAS; API never proxies bytes)
- client `PUT`s bytes to `upload_url` with matching `Content-Type`
- `POST /v1/kb/uploads/{id}/complete` → server HEADs the blob, flips `initiated → uploaded` (missing → `upload_incomplete`, oversize → `size_exceeded`, type mismatch → `type_mismatch`, empty → `empty_content`)
- worker drains `uploaded → chunking → embedding → finished` (`embed_failed`/`persist_failed`/`decode_failed` on failure)
- `GET /v1/kb/documents/{id}` → `{document_id, status, chunk_count, error_code}` at any time

Default LLM transport points at Ollama (`http://localhost:11434/v1`); swap `OPENAI_BASE_URL` to any OpenAI-compatible provider.

## Layout

- `src/rag_lab/core/` — `Settings`, Pydantic schemas (`Document`, `Chunk`, `RetrievalResult`, `AnswerWithCitations`, `Trace`)
- `src/rag_lab/adapters/` — `OpenAICompatibleLlm` (chat), `EmbeddingPort` (OpenAI-compatible + stub fake), `StoragePort` (signed-URL issue + stat/download, Azurite + in-memory fake), `DocumentStorePort` (async lifecycle: initiate → uploaded → chunking → embedding → finished/failed; Postgres + in-memory fake), `db_ping`
- `src/rag_lab/ingest/` — `LoaderPort`, `TextLoader`, `PdfLoader`, `split_chunks` (heading-aware, overlap windows)
- `src/rag_lab/worker/` — `pipeline.drain/process_one` (claim → download → load → split → embed → finish), `run` entrypoint (`python -m rag_lab.worker.run`)
- `compose.yaml` — `db` (pgvector:pg16, :5432), `azurite` (:10000-10002), `worker` (ingest poll loop)
