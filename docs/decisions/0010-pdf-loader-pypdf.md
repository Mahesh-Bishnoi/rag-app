# ADR-0010: PDF loader via pypdf (per-page chunks, text-only v1)

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-004 promises PDF + text in v1 behind one `LoaderPort`; ADR-0009 left
  `POST /v1/kb/upload` returning `415` for PDFs until a loader registers.
- Freshness check 2026-10-11 (PyPI): `pypdf 6.20.0`, `pdfplumber 0.11.10`,
  `PyMuPDF 1.28.2`. Box is Ryzen 7 5800H + RTX 3050 Mobile + ~13.5GB RAM,
  podman-first; v1 PDF need is text + page numbers for structure-aware
  chunking, not tables/OCR.
- Upload hardening already in place (`ALLOWED_CONTENT_TYPES`, 20MB cap,
  filename sanitization, generic 415/422/500s per ADR-0005/0009).

## Decision
1. Add `pypdf>=6.20` (pinned by `uv.lock`; bump process per ADR-0006:
   check PyPI latest, update floor, `uv sync`, verify gates, record here
   or in a superseding ADR).
2. New `PdfLoader` in `src/rag_lab/ingest/loaders.py` implementing
   `LoaderPort`: `can_handle` on `.pdf` suffix or
   `application/pdf` content type; `load()` extracts text per page via
   `PdfReader`, one `Chunk` per non-empty page (`page` 1-indexed,
   `mime="application/pdf"`, `byte_start/byte_end=None` — byte offsets
   are meaningless for PDF content streams).
3. Parse failures (truncated bytes, non-PDF masquerading as PDF,
   encrypted without password) raise `ValueError`, which the existing
   `_load_chunks` maps to generic `422`. Empty-after-extract (e.g.
   scanned image PDF) returns `[]`, also `422` — honest, no empty ingest.
4. Register `PdfLoader()` alongside `TextLoader()` in `create_app`;
   no change to upload orchestration (`_upload_file` dispatch unchanged).

## Alternatives Considered

### pdfplumber (0.11.10)
- Pros: Better tables/layout via pdfminer, useful for Excel-like follow-ups
- Cons: Heavier dep tree (pdfminer.six, Pillow); layout fidelity unused by
  v1 per-page text chunks; more RAM/CPU on a constrained laptop
- Rejected for v1: Pay for tables when a table-aware slice needs it

### PyMuPDF (1.28.2)
- Pros: Fastest extraction, best layout/muPDF engine
- Cons: AGPL/commercial dual license friction; native binary wheel;
  speed unneeded at v1 upload volumes
- Rejected: License + binary weight outweigh speed for a learning lab

### OCR/vision PDFs in v1 (ocrmypdf, pytesseract)
- Pros: Scanned PDFs ingest instead of `422`
- Cons: New system deps (tesseract, OCR models), GPU/RAM pressure,
  whole new eval surface
- Rejected: Out of scope per intent (PDF+text v1); `422` stays honest

## Consequences
- `POST /v1/kb/upload` accepts valid PDFs; `test_upload_pdf_without_loader_is_415`
  becomes a valid-PDF round-trip plus invalid-PDF `422` coverage.
- Decompression-bomb / giant-page-count PDFs bounded only by the existing
  20MB `max_upload_bytes` cap (accepted lab limit; a page-count cap is
  the later fix if it hurts).
- DB failure after blob put can still orphan a blob (accepted limit from
  ADR-0009; unchanged).
- Structure-aware heading splits inside a page stay a later chunking-slice
  concern; this slice proves pages only.
