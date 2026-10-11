# ADR-004: V1 ingests PDF + text only, behind a modular loader interface

## Status
Accepted

## Date
2026-10-11

## Context
User listed PDF, Excel, Word, text, image, website links, then agreed v1 proves the loop with PDF + text while design stays modular for the rest. Each deferred type is a different problem: Excel needs table-aware chunking, Word needs docx extraction, images need OCR/vision, links need crawling + freshness.

## Decision
V1 ships PDF + plain text / markdown loaders only. Every loader implements one interface: detect → extract (text + structure: title, heading path, pages) → emit chunks with metadata. Excel/Word/image/link loaders come later without changing the retrieve/rerank/generate path.

## Alternatives Considered

### All six types in v1
- Pros: Single KB that eats anything from day one
- Cons: Parser sprawl buries the core chunk → embed → retrieve → cite loop; blows up eval surface; risks never finishing the learning loop
- Rejected: Too big for a fundamentals build

### Text only in v1
- Pros: Fastest loop
- Cons: PDF structure (headings, tables, pages) is where structure-aware chunking actually teaches something; dropping it weakens the chunking lesson
- Rejected: PDF is worth the extra parser

## Consequences
- Loader interface is a one-way door — get the chunk+metadata shape right now (source, title, heading path, page, mime, byte offsets)
- Deferred types tracked as follow-ups, not v1 acceptance criteria
- Eval set built on PDF + text only for v1
