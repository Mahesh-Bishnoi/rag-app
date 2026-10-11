# ADR-002: Podman with Postgres+pgvector and S3-compatible store

## Status
Accepted

## Date
2026-10-11

## Context
Need local services for vectors, metadata, and raw file bytes. User explicitly wants podman for databases (postgres+pgvector), S3-type bucket, services. No scaling/uptime SLAs. Must fit in ~13.5GB RAM alongside a small local model.

## Decision
Podman Compose runs postgres+pgvector (vectors + metadata + BM25 path) and an S3-compatible object store (raw uploads: PDF, text; later Excel/Word/image). App connects over local ports; data volumes persist.

## Alternatives Considered

### Embedded vector lib only (Chroma / LanceDB in-process)
- Pros: Zero services, simplest start
- Cons: Ignores user's explicit podman + postgres request; weaker story for metadata filters, persistence, and later multi-type growth
- Rejected: Conflicts with stated constraint

### Milvus + Elasticsearch (blueprint default)
- Pros: Matches blueprint scale, best hybrid story at enterprise size
- Cons: Heavy RAM/ops for this box; two systems to learn instead of RAG itself
- Rejected: Oversized for v1 learning goal

## Consequences
- Compose file becomes a one-way door: service names, ports, volume layout must stay stable
- Postgres schema must carry chunk text, embeddings, document metadata (title, heading path, source, version) from day one for hybrid + filtered retrieval
- S3 store holds originals; DB holds derived chunks — no raw bytes in Postgres
