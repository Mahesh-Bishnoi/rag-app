# Intent: Local RAG Lab

## Outcome
Local RAG playground to learn fundamentals by reading code, starting from NVIDIA blueprint ideas but lightweight enough for a laptop with a small GPU.

## User
Developer (self) learning RAG internals and the Python stack, while testing repo skills (clean-code primary, adr, debug).

## Why now
RAG details unknown; NVIDIA blueprint looked like a good starting point but needs validation against current practice. Good opportunity to learn Python + Pydantic + PydanticAI + FastAPI + LangChain + LangGraph on a real build.

## Success
- `podman compose up` brings up postgres+pgvector, S3-type store, services
- Drag-drop PDF + text into a knowledge base, chat over them with citations
- Visible pipeline: ingest → structure-aware chunking → BM25+vector hybrid → rerank → generate
- RAGAS-style eval + basic observability (traces, retrieval scores)
- Code satisfies clean-code skill; decisions recorded as ADRs; debuggable from day one

## Constraint
- Machine: Ryzen 7 5800H + RTX 3050 Mobile + ~13.5GB RAM — size models/services to fit
- Podman-first for services
- Stack: Python, Pydantic, PydanticAI, FastAPI, LangChain, LangGraph
- Modular loader design so Excel / Word / image / website links can be added later
- UI/UX + CSS skills low priority for v1

## Out of scope
- Multi-user auth, scaling, uptime SLAs
- Full NVIDIA NIM / Helm blueprint deploy (too heavy for this box; use as concept reference only)
- Full 6-type ingestion in v1 (PDF + text first)
