# ADR-0017: Minimal drag-drop UI served from FastAPI

## Status
Accepted

## Date
2026-10-11

## Context
- Intent promises drag-drop PDF + text into a knowledge base and chat with
  citations. The backend flow exists (`POST /v1/kb/uploads/initiate` → client
  `PUT` → `POST .../complete` → poll `GET /v1/kb/documents/{id}` → `POST
  /v1/chat/completions` + `GET /v1/traces/{id}`), but there is no page driving
  it — verification means hand-rolled `curl`.
- UI/CSS skills are low priority for v1; the page is a lab tool, not a product.
- FastAPI 0.143.0 (installed) offers `app.frontend()` for built SPAs.

## Decision
1. Single dependency-free page at `GET /`: `src/rag_lab/api/static/index.html`
   (drop zone + file picker, per-file ingest status with 60s poll, chat form,
   answer + citations + trace link, health line). Inline CSS/JS, no build step.
2. New `rag_lab/api/ui.py` owns the bytes (`index_page() -> HTMLResponse`,
   read off the packaged static file); `create_app()` adds one explicit
   `GET /` route with `include_in_schema=False`. No new dependencies.
3. Browser JS renders all server/LLM data via `textContent` + `createElement`
   (never `innerHTML`); same-origin `fetch` only; no tokens stored. Upload
   validation stays server-side (MIME allowlist, size cap, filename sanitize);
   the page only mirrors the allowlist for early rejection.
4. Explicit route instead of `app.frontend()`: the SPA fallback would also
   answer unknown paths, shadowing API 404s. `tests/test_ui.py` pins `GET /`
   as HTML, unknown `/v1/*` as 404, and the absence of `innerHTML`.

## Alternatives Considered

### `app.frontend()` with a static directory
- Pros: Vendor-skill default; free SPA fallback
- Cons: Fallback answers unknown API paths with HTML instead of 404; implies
  a build output directory that does not exist
- Rejected: explicit single route keeps 404s honest

### Separate frontend service (Vite/React)
- Pros: Real component model, nicer styling
- Cons: New toolchain, second serve path, container wiring — before any page
  proves the flow; UI is low priority for v1
- Rejected: dependency-free page proves the flow; revisit with a stated reason

### Inline HTML string in `main.py`
- Pros: Zero new files
- Cons: Presentation bloats the API entrypoint (already ~440 lines); harder
  to review JS/CSS diffs
- Rejected: `ui.py` + static file keeps one owner per concern

## Consequences
- `fastapi dev` serves the lab UI at `http://localhost:8000/` with zero extra
  steps; README documents the flow.
- Hermetic tests cover the route; `mypy strict`, ruff set, `lint-imports`
  (same `rag_lab.api` layer), and `deptry` stay green — no new imports beyond
  `fastapi.responses` + stdlib `pathlib`.
- Known limits: single-user lab page (no auth — out of scope per intent); no
  deletion/re-ingest controls; 60s ingest poll; chat uses the first
  `/v1/models` entry. Upgrades get their own ADR.
