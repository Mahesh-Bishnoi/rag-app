# ADR-0018: Browser uploads need emulator CORS + `x-ms-blob-type`

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-0017's page `PUT`s bytes straight from the browser to the Azurite URL
  from `POST /v1/kb/uploads/initiate`. On a fresh machine (`podman compose up`
  after `compose down -v`) every upload failed in preflight with the quoted
  symptom: `403 CORS not enabled or no matching rule found for this request`.
- Debug loop (`OPTIONS` preflight + `PUT` against Azurite, no secrets printed)
  showed the service had **zero** CORS rules; writing one rule flipped
  preflight `403 → 200`. Rules survive `podman restart` but not volume
  deletion, so a manual one-off fix does not survive onboarding.
- The same loop exposed a second, independent failure: a `PUT` without
  `x-ms-blob-type` answers `400`; with `x-ms-blob-type: BlockBlob` it lands
  (`201`). The SDK path already knew this (`tests/test_adapters.py`); only
  the browser UI missed the header.

## Decision
1. API startup ensures one CORS rule on the **emulator only**
   (`AzuriteBlobStorage.ensure_cors`, called fail-open from a `_lifespan` in
   `main.py`): origins from `Settings.blob_cors_origins`
   (`http://localhost:8000` + `http://127.0.0.1:8000`), methods
   `PUT,GET,HEAD,OPTIONS`, `allowed_headers: ["*"]` (covers `Content-Type`
   and `x-ms-blob-type`), `max_age_in_seconds: 3600`. Existing rules are
   preserved; a matching rule short-circuits (idempotent).
2. Emulator guard: rules are written only when the connection string parses
   to `AccountName=devstoreaccount1` (`DEV_EMULATOR_ACCOUNT`). Any other
   account returns `False` before any IO — real Azure keeps operator-owned
   CORS; the app must not overwrite it. Emulator IO failures raise; the
   lifespan catches and logs so the API still serves (`GET /` → 200 with
   Azurite down).
3. The UI `PUT` sends `x-ms-blob-type: BlockBlob` alongside `Content-Type`.
4. Regression cover: `tests/test_storage_cors.py` (rule shape, origin
   mismatch, non-emulator skip without IO, live idempotent round-trip) and
   `tests/test_ui.py` (header present in served page, lifespan runs with
   memory storage, lifespan fails open against a closed port).

## Alternatives Considered

### Manual CORS setup script / documented `az` command
- Pros: No app code; operator owns emulator config
- Cons: Every fresh volume re-breaks uploads; onboarding step nobody
  remembers; the failure (browser preflight 403) points nowhere near the fix
- Rejected: startup ensure is idempotent and self-healing

### `allowed_origins: ["*"]`
- Pros: Covers any dev host/port
- Cons: Over-broad even for a local emulator; hides origin misconfiguration
  the explicit list catches
- Rejected: two explicit origins, configurable via `blob_cors_origins`

### Same ensure against real Azure
- Pros: One code path everywhere
- Cons: App overwrites operator-owned CORS on every boot — a production
  footgun with no rollback
- Rejected: hard guard on the well-known emulator account

## Consequences
- Fresh `compose down -v` → `compose up` → `fastapi dev` uploads work with
  zero extra steps; verified from-zero (wipe rules → loop RED with the exact
  symptom → app boot → loop GREEN: preflight 200, PUT 201; second boot keeps
  exactly 1 rule).
- `CorsRule` fields round-trip from the service as comma-joined strings, not
  lists — `cors_rule_matches` normalises both forms (`_csv`); tests pin it.
- SDK clients pass `retry_total=0` everywhere including the CORS path, so a
  down Azurite fails fast instead of backing off (matters for `/health` and
  startup).
- Known limits: no CORS support for non-emulator accounts by design; UI still
  single-user lab page per ADR-0017.
