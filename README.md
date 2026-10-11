# rag-lab

Local RAG lab (infra+skeleton v1). See `docs/intent/rag-lab.md` and `docs/decisions/`.

## Quick start

1. Copy env: `cp .env.example .env`
2. Install: `uv sync --group dev`
3. Infra: `podman compose up -d`
4. Schema (fresh volume): `uv run alembic upgrade head`
5. Run API (dev): `uv run fastapi dev` (reload, entrypoint in `[tool.fastapi]`);
   prod: `uv run fastapi run`
6. Checks: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`,
   `uv run mypy .` (strict), `PYTHONPATH=src uv run lint-imports`,
   `uv run deptry src alembic`, `uv audit`
7. Secrets: `gitleaks detect --source . --config .gitleaks.toml` (clean required;
   allowlist in `.gitleaks.toml` covers only well-known non-secrets, with reason)
7. Live tier (needs `podman compose up -d` + `.env`):
   `RAGLAB_LIVE_DB=1 uv run pytest` — migration upgrade/downgrade round-trip
   plus Azurite put round-trip. Focused runs skip the coverage gate with
   `uv run pytest <path> --no-cov`.

Existing dev database predates Alembic (built by the skeleton init SQL):
`uv run alembic stamp head` baselines it without re-running migration `0001`.
New migrations: `uv run alembic revision -m "..." --autogenerate` (once ORM
models land and attach to `target_metadata` in `alembic/env.py`).

## API (OpenAI-compatible + KB extension)

- `GET /health` → `{status, db, blob}`
- `GET /v1/models` → configured chat + embedding model ids
- `POST /v1/chat/completions` → `501` until Phase 3 wires retrieve → rerank → generate
- `POST /v1/kb/upload` (multipart `file`; text/pdf only, 20MB cap) → `{name, blob_url, bytes, document_id, chunk_count}` (PDF returns `415` until the PDF-loader slice lands)

Default LLM transport points at Ollama (`http://localhost:11434/v1`); swap `OPENAI_BASE_URL` to any OpenAI-compatible provider.

## Layout

- `src/rag_lab/core/` — `Settings`, Pydantic schemas (`Document`, `Chunk`, `RetrievalResult`, `AnswerWithCitations`, `Trace`)
- `src/rag_lab/adapters/` — `OpenAICompatibleLlm`, `StoragePort` (Azurite + in-memory fake), `db_ping`
- `src/rag_lab/ingest/` — `LoaderPort`, `TextLoader` (PDF next)
- `compose.yaml` — `db` (pgvector:pg16, :5432), `azurite` (:10000-10002)
