# ADR-0007: Adopt FastAPI vendor skill; keep SQLAlchemy over SQLModel

## Status
Accepted

## Date
2026-10-11

## Context
- FastAPI ships an official AI-agent skill, version-matched to the installed
  package (verified: skill 0.143.0 against FastAPI 0.143.0). Installed via
  `uvx library-skills install -s fastapi` as a repo-local symlink in
  `.agents/skills/fastapi`, so guidance tracks `uv.lock` upgrades.
- The skill's conventions mostly match our code (sync `def` for blocking
  paths, one operation per function, no ellipsis/`RootModel`, no
  ORJSON/UJSON). Three items needed action; one conflicts with a prior ADR.

## Decision
1. `Annotated[UploadFile, File()]` param style; Pydantic models as endpoint
   return types (`HealthResponse`, `ModelsResponse`, `UploadResponse`)
   instead of `dict[str, Any]`. The `B008` noqa this removed is deleted.
2. Ruff `FAST` family enabled (FAST001–003); `fastapi-cli` added with
   `[tool.fastapi] entrypoint = "rag_lab.api.main:app"`. Dev runs
   `uv run fastapi dev`, prod runs `uv run fastapi run`; both verified
   against live services.
3. **Reject the skill's SQLModel-over-SQLAlchemy preference.** ADR-0006
   chose SQLAlchemy 2.1 + Alembic for versioned migrations; SQLModel would
   trade that away for model-sharing convenience we don't need (no ORM
   models exist yet, and request/response schemas stay separate per the
   skill's own response guidance). Project ADR wins over vendor default
   here. Revisit only if ORM models and API schemas genuinely converge.

## Alternatives Considered

### SQLModel now (follow the skill verbatim)
- Pros: Single model class for table + validation; vendor-blessed
- Cons: Reverses ADR-0006's migration story six hours after recording it;
  couples storage schema to API schema, which the skill itself warns
  against via separate response models
- Rejected: No evidence the convergence pays for this project

### Stay on raw uvicorn commands, skip fastapi-cli
- Pros: One fewer dependency
- Cons: Diverges from the vendor workflow the skill prescribes; loses
  `fastapi dev` reload + `fastapi run` production defaults
- Rejected: Small dep, verified working with our src layout

## Consequences
- `.agents/skills/fastapi` symlink must be committed so the skill resolves
  after install; re-run `uvx library-skills install -s fastapi` if it dangles.
- `fastapi-cli` joins the deptry CLI-used ignore list with reason.
- Future skill updates arrive via `uv sync` (package bump), not manual edits.
