# ADR-001: Build lightweight local RAG instead of deploying NVIDIA blueprint as-is

## Status
Accepted

## Date
2026-10-11

## Context
NVIDIA rag-blueprint skill v2.6.0 targets Docker/Compose or Kubernetes/Helm with GPU NIM microservices (LLM, embedding, ranking, VLM, OCR, parse), Milvus/Elasticsearch, SeaweedFS, guardrails, observability. Machine is Ryzen 7 5800H + RTX 3050 Mobile + ~13.5GB RAM, podman-first, local-only. Goal is learning fundamentals by reading code, not enterprise deploy. 2026 consensus techniques to keep: structure-aware chunking, BM25+vector hybrid + reciprocal-rank fusion, cross-encoder rerank, RAGAS eval, citations, traces.

## Decision
Use the blueprint as a concept map (ingest, search-and-retrieval, reasoning-and-generation, eval, observability categories) and build a small local app sized to the box.

## Alternatives Considered

### Deploy NVIDIA-AI-Blueprints/rag via Compose/Helm with self-hosted NIMs
- Pros: Full features (VLM, guardrails, agentic RAG), pre-validated
- Cons: GPU/VRAM, RAM, and ops weight far exceed RTX 3050 Mobile + 13.5GB; hides fundamentals behind services; conflicts with local learning goal
- Rejected: Right reference, wrong runtime for v1

### NVIDIA-hosted NIM endpoints + local orchestration
- Pros: Less local GPU pressure, still uses blueprint services
- Cons: Needs NGC key, network dependency, cost; violates local-only constraint; still hides embedding/rerank internals
- Rejected: Breaks local-only learning constraint

## Consequences
- Must re-implement core stages simply (ingest, chunk, embed, hybrid retrieve, rerank, generate) — that is the learning payoff
- Keep blueprint reference files as background reading, not runtime dependency
- Revisit NIM/Helm only if/when a bigger box or hosted need appears
