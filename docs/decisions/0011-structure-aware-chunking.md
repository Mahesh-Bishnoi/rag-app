# ADR-0011: Structure-aware chunking after load (headings, overlap windows)

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-004 promises structure-aware chunking (headings, pages, sections with
  title + heading path per chunk); loaders so far emit one chunk per file
  (text) or per page (PDF) with `heading_path=""`.
- Upload already orchestrates validate → load → blob put → persist
  (ADR-0009). Long pages/files would embed and retrieve as oversized units.
- Constraints: pure logic in `ingest`, layers `api → adapters/ingest → core`
  kept; hermetic tests by default; ruff complexity ≤10; mypy strict.

## Decision
1. New pure `ingest/chunking.py`: `split_chunks(chunks,
   max_chars=1500, overlap=200)` — no IO, no config lookup.
2. Heading pass splits markdown `#`–`######` sections (`# tag` without a
   space stays body text); `heading_path` joins the stack (`Alpha > Beta`);
   heading-only sections with no body are dropped. Non-markdown keeps the
   loader's `heading_path`.
3. Long sections split on blank lines, then sentence boundaries, then hard
   cuts; pieces pack to `max_chars`; overlap prepends up to `overlap` tail
   chars of the previous piece, shrunk to fit so no piece exceeds
   `max_chars`. Each input page splits independently, keeping `page`/`mime`.
4. Byte offsets survive only when a chunk passes through unsplit; split
   pieces get `None` (byte offsets are meaningless after re-chunking).
5. `POST /v1/kb/upload` applies `split_chunks` after `load()` and before blob
   put; empty-after-split returns `422` (same honest rule as image-only PDFs).

## Alternatives Considered

### Token-based splitter (tiktoken, LangChain splitters)
- Pros: Model-accurate budgets; LangChain is the anticipated splitter seam
  (ADR-003)
- Cons: New dependency + tokenizer data for a v1 char budget that already
  unblocks embed/retrieve; LangChain seam stays open for later
- Rejected for this slice: char windows prove the pipeline; swap when a
  model-sensitive slice needs it

### Chunk inside each loader
- Pros: Loader owns its structure (PDF pages vs markdown headings)
- Cons: Overlap/window logic duplicates per loader; upload orchestrates two
  different shapes
- Rejected: one post-load pass keeps loaders small and overlap in one place

### Config-driven sizes in Settings now
- Pros: Tunable without code change
- Cons: No retrieval data yet to tune against; extra knobs before first
  measurement
- Rejected (YAGNI): module defaults; promote to `Settings` when eval
  justifies tuning

## Consequences
- Upload `chunk_count` grows for long files; existing short-file round-trips
  unchanged (passthrough preserves offsets).
- `PostgresDocumentStore` needs no change (same `save(document, chunks)`).
- Token accuracy and per-page heading splits inside scanned layouts stay
  later concerns; the shortcut is recorded in code only by the `None`
  offsets on split pieces.
- Live proof stays hermetic (`test_chunking.py` + upload heading test); no
  migration (no schema change).
