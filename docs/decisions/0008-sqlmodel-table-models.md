# ADR-0008: SQLModel for table models (supersedes ADR-0007 §3)

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-0006 chose SQLAlchemy 2.1 + Alembic; ADR-0007 rejected the FastAPI
  skill's SQLModel preference to protect the migration story and keep
  storage and API schemas decoupled.
- The user then directed the switch to SQLModel. Verification first:
  SQLModel 0.0.48 (Oct 6, 2026) supports SQLAlchemy 2.1 — resolver proves
  `sqlmodel 0.0.48 + pgvector 0.5.1` coexist with our `sqlalchemy 2.1.4`,
  so no 2.0 downgrade. SQLModel is pre-1.0 (accepted risk, pinned by lockfile).
- Key correction to the ADR-0007 framing: SQLModel replaces the *model
  layer*, not the engine or migrations. Alembic stays and gets strictly
  better: `target_metadata = SQLModel.metadata` completes the autogenerate
  wiring ADR-0006 left as `None`.

## Decision
1. Table models are SQLModel (`src/rag_lab/adapters/tables.py`):
   `DocumentRow` / `ChunkRow` mirror migration 0001 exactly (UUID PKs with
   `gen_random_uuid()`, `Vector(768)`, persisted `tsvector` computed column,
   same GIN indexes). Required the `pgvector` pip package earlier than
   planned — it was already the anticipated Phase 2 dependency.
2. The ADR-0007 decoupling concern stands as a discipline rule, not a
   framework choice: table rows are never returned from endpoints, Pydantic
   DTOs are never persisted, no sharing/subclassing/implicit conversion
   between them. They change for different reasons.
3. Parity is tested, not assumed: hermetic metadata contract
   (`test_tables.py`) plus live reflection of the migrated schema against
   metadata column sets inside the migration round-trip.
4. Engine helper (`get_engine`, `postgresql+psycopg`) is unchanged —
   SQLModel works with any SQLAlchemy engine.

## Alternatives Considered

### Stay on plain SQLAlchemy (ADR-0006/0007 direction)
- Pros: No 0.0.x dependency; keeps vendor at arm's length
- Cons: Direct user direction otherwise; hand-rolled models plus later
  autogenerate wiring done twice; diverges from the FastAPI-stack
  coherence the skill prescribes
- Rejected: User decision; switch cost is near-zero today (no ORM models
  existed) and grows with every future table

### SQLModel for tables AND API schemas (full convergence)
- Pros: Fewest classes
- Cons: Couples storage schema to API stability — the exact failure the
  skill's own separate-response-model guidance warns against
- Rejected: Two-model discipline above

## Consequences
- Autogenerate noise now; the next real model change must calibrate
  autogenerate output (computed/GIN rendering) before trusting it blindly.
- SQLModel 0.0.x upgrades get the ADR-0006 bump treatment: pull, verify
  gates, record.
- AGENTS.md quality bars updated to name SQLModel as the table-model layer.
