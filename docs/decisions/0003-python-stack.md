# ADR-003: Python stack — FastAPI + Pydantic + PydanticAI + LangChain + LangGraph

## Status
Accepted

## Date
2026-10-11

## Context
User wants to learn this specific stack on a real build: Python, Pydantic, PydanticAI, FastAPI, LangChain, LangGraph. Clean-code skill is primary; adr + debug also exercised; UI/CSS low priority. Risk is overlap: PydanticAI vs LangChain/LangGraph can duplicate agent/graph duties.

## Decision
FastAPI serves API + drag-drop KB + chat endpoints. Pydantic models all IO (ingest, chunk, retrieval, generation). PydanticAI for typed agent/generation layer. LangChain for document loaders/text splitters/integrations where they save parser work. LangGraph for the explicit retrieve → rerank → generate pipeline graph so the learning loop is visible and traceable.

## Alternatives Considered

### FastAPI + Pydantic only, no LangChain/LangGraph
- Pros: Smallest surface, cleanest code, no framework overlap
- Cons: Misses user's explicit learning goal for LangChain/LangGraph; hand-rolls splitters/integrations
- Rejected: Violates learning intent

### PydanticAI only for orchestration (skip LangGraph)
- Pros: Less duplication, single agent abstraction
- Cons: Loses the explicit graph visualization of the RAG stages the user wants to learn
- Rejected: Graph visibility is part of success

## Consequences
- Must draw a hard seam: LangGraph owns pipeline flow, PydanticAI owns typed LLM calls, LangChain owns loaders/splitters — no cross-use to avoid two ways to do the same step
- All stage boundaries get Pydantic schemas so eval/observability can log them
- Bigger dependency surface; lock versions early
