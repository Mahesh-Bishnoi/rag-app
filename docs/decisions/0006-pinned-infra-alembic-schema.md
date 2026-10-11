# ADR-0006: Digest-pinned infra, PG16 stay, Alembic owns schema

## Status
Accepted

## Date
2026-10-11

## Context
- Online check 2026-10-11: Azurite `3.37.0` (Aug 2026) is latest — matches the cached/running image. pgvector extension `0.8.7` is latest — matches what the `pg16` rolling tag ships (verified `extversion 0.8.7` in the dev DB). Python side: SQLAlchemy `2.1.4` (2026-10-07) and Alembic `1.20.0` (2026-09-11) are latest.
- Upstream also ships pgvector for PG 17/18 and official Postgres is at 18.6, but a PG major jump needs dump/reload.
- Skeleton phase created schema via `db/init/001_schema.sql` mounted into the container entrypoint. That path only runs on first volume init, is invisible to code review of schema evolution, and would drift from any second schema path — two sources of truth.

## Decision
1. Stay on Postgres 16 (`pgvector/pgvector:pg16`) for v1. PG 17/18 migration is a deferred follow-up, not v1 criteria.
2. Pin both compose images by digest (same binaries as verified running):
   - `pgvector/pgvector:pg16@sha256:a97d77306ff47cc1d8add4d000cec2207da2e110746318df3137ce53c337ca30`
   - `mcr.microsoft.com/azure-storage/azurite:latest@sha256:f9791d0a0613e93057c06ae31f7ac0c0222f0530590e67e020b0986c66a36fa6`
   Bump process: `podman pull <tag>`, read `.Digest` via `podman image inspect`, update `compose.yaml`, `podman compose up -d`, note it here or in a superseding ADR.
3. Add `sqlalchemy>=2.1` + `alembic>=1.20` (resolved 2.1.4 / 1.20.0). Engine uses the `postgresql+psycopg` driver over the existing `psycopg[binary]` install.
4. **Alembic owns the schema going forward.** Baseline migration `0001` carries extensions (`vector`, `pg_trgm`) + `documents` + `chunks`, in the exact SQL proven by the skeleton phase. `db/init/001_schema.sql` and its compose mount are deleted — no second path. Fresh setup is `podman compose up -d && uv run alembic upgrade head`.
5. The dev database (already matching the baseline via the old init path) is moved under Alembic control with `alembic stamp head`, not by re-running the migration.

## Alternatives Considered

### Keep db/init SQL and skip Alembic
- Pros: No new deps, `compose up` alone yields schema
- Cons: No versioned evolution; every schema change becomes a hand-managed init edit that never runs on existing volumes; contradicts the "debuggable, traceable" success criteria
- Rejected: Migrations are the mechanism that scales past v1

### Keep both init SQL and Alembic
- Pros: `compose up` works standalone
- Cons: Two schema sources that will drift; a fresh volume and a migrated volume can silently differ
- Rejected: One source of truth (clean-code locality)

### SQLAlchemy 2.0.x (older stable) instead of 2.1.x
- Pros: Longer track record; 2.1 is only ~3 weeks post-GA
- Cons: 2.1 is the current default `pip install sqlalchemy` series; pinning old defers a migration that only gets harder with ORM models in place
- Rejected: Adopt current at the start when the migration surface is zero ORM models

### Jump to PG 17/18 now
- Pros: Newest server features, longest support runway
- Cons: Dump/reload cycle plus re-verification of pg_trgm/vector behavior for zero v1 learning payoff; PG16 is supported and already proven on this box
- Rejected for v1: Deferred follow-up

## Consequences
- `env.py` reads `DATABASE_URL` from `Settings`; `target_metadata` stays `None` until the first ORM models land (autogenerate comes then, not now).
- Baseline migration uses raw SQL (no `pgvector` pip package yet); the Python `pgvector` integration lands with Phase 2 embedding writes.
- Digest pins freeze binaries: upstream patches no longer flow in silently — each bump is a deliberate, recorded act.
- README documents the two-command fresh setup and the stamp-vs-upgrade distinction.
