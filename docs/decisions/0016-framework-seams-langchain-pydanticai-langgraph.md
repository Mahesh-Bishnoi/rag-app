# ADR-0016: Framework seams — LangChain splitters, PydanticAI generation, LangGraph flow

## Status
Accepted

## Date
2026-10-11

## Context
- ADR-0003 decided the stack seam (LangGraph = pipeline flow, PydanticAI =
  typed LLM calls, LangChain = loaders/splitters) but no framework code
  existed: zero `langchain|langgraph|pydantic_ai` references in `src/`, and
  none of the three in `pyproject.toml`. Ingest splitting, generation, and
  both pipeline flows were hand-rolled.
- Scope answers (user, 2026-10-11): (1) minimal LangChain surface
  (`langchain-text-splitters` only, no `langchain-community`); (2) new
  `rag_lab.graph` layer between `worker` and `adapters` (explicit seam over
  stuffing orchestration into `adapters`); (3) graphs for both ingest and QA.
- Freshness check (PyPI, 2026-10-11): `langchain-text-splitters 1.1.3`,
  `langgraph 1.2.14`, `pydantic-ai 2.55.0`. Pinned as lower bounds in
  `pyproject.toml`, frozen in `uv.lock`. Bump process: re-check PyPI latest
  at adoption time, raise the bound, re-lock, record the versions here.
  (`langchain-core` was added then removed in the same change — nothing
  imports it directly; it stays a transitive pin via `langgraph`.)

## Decision
1. LangChain owns splitters: `ingest/chunking.py` keeps the custom markdown
   heading sectioning (`_sections`) and delegates long-section windowing to
   `RecursiveCharacterTextSplitter(chunk_size=max_chars,
   chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""])`. Empty pieces
   are filtered so the no-empty-chunks contract holds. `PdfLoader` keeps
   `pypdf` under the hood (no community loaders — minimal surface).
2. PydanticAI owns typed generation: new `adapters/generation.py` with
   `GroundedAnswer(answer: str, min_length=1)`, the single-owner
   `grounded_prompt()`, and `complete()`/`generate_answer()` that run a
   PydanticAI `Agent(output_type=GroundedAnswer)` over a `FunctionModel`
   wrapping the injected `LlmPort` — exactly the direction
   `adapters/llm_openai.py` prescribed ("the PydanticAI generation layer
   calls through this port"). `StubLlm` keeps hermetic tests offline; an
   empty answer fails validation (`UnexpectedModelBehavior`) and surfaces as
   a generation error, never a silent 200.
3. LangGraph owns flow: new `rag_lab/graph/` package with `qa.py`
   (embed → retrieve → rerank → generate, `QaState`/`QaOutcome` TypedDicts)
   and `ingest.py` (download → load → split → embed → finish, `_IngestFailure`
   maps to the unchanged `doc_store.fail` codes). States are `total=False`
   because graph input is always a key subset. Stage commits keep their order
   so crash-retry idempotency is unchanged.
4. Error contract preserved: the generate node wraps failures in
   `QaGenerationError` so the API keeps distinct 500s (`retrieval failed` vs
   `generation failed`) with the same log lines. Trace save stays fail-open.
5. `worker/pipeline.py` thins to `process_one`/`drain` over
   `run_ingest_one`; `api/main.py::_answer_question` runs `run_qa` then owns
   trace/logging/citations as before. `blob_name` moves to
   `adapters/storage.py` (it inverts `StoragePort.blob_url`); callers updated.
6. `.importlinter` gains the `rag_lab.graph` layer between `worker` and
   `adapters` (reorder, not loosen): `api`/`worker` → `graph` →
   `adapters`/`ingest` → `core`.

## Alternatives Considered

### Keep seams inside `rag_lab/adapters` (no new layer)
- Pros: Zero `.importlinter` change; fakes next to prod adapters
- Cons: The orchestrator lives inside the orchestrated layer; `adapters/`
  mixes transports, orchestration, and generation
- Rejected: user decision; the visible layer is the learning goal (ADR-0003)

### PydanticAI calling the provider directly (new model path)
- Pros: Uses PydanticAI's OpenAI/Ollama models end to end
- Cons: Second LLM transport beside `LlmPort`; hermetic tests need a new
  fake; bypasses the prescribed port direction
- Rejected: `FunctionModel`-over-`LlmPort` keeps one transport seam

### Conditional-error edges in the ingest graph
- Pros: More "graphy" — each failure an explicit edge to a fail node
- Cons: Spreads one abort concept across five edges; the linear chain with a
  typed exception is the smallest complete expression
- Rejected for v1: linear chain + `_IngestFailure`

### `langchain-community` loaders now
- Pros: More loader coverage sooner
- Cons: Large new surface before the seam proves itself; laptop budget
- Rejected: minimal surface per user decision; community loaders arrive with
  the Excel/Word/image workstream, behind the same `LoaderPort`

## Consequences
- `tests/test_graph.py` covers prompt shape, typed generation (+ loud empty
  answers), QA happy/empty/failure paths, and ingest claim→fail mapping;
  existing chat/worker/chunking tests pass unchanged through the graphs
  (splitter parity held: same chunk counts, heading paths, overlap).
- Hermetic coverage stays above the 80 floor; `mypy strict`, the ruff set,
  `lint-imports`, and `deptry` stay green. `CompiledStateGraph` returns are
  pinned as `CompiledStateGraph[State, None, State, State]` (the inferred
  shape); node funcs stay plain `def`s because mypy only infers node input
  types from declarations, not from `Callable`-returning factories.
- Queries + answers keep flowing through the trace store unchanged; no new
  retention questions (still owned by the future ADR flagged in ADR-0015).
- Live-tier drive-by fix (pre-existing, found while proving this change):
  `PostgresTraceStore._hydrate` built `Chunk(content="")` for stored ids with
  no chunk row, violating `Chunk.content`'s `min_length=1` and failing
  `test_postgres_trace_save_get_round_trip_with_hydration` on its second
  save/get. The fallback now yields `"(chunk content unavailable)"` so a
  trace keeps its id + scores even after its chunk rows are gone.
- A model with dim != 768 still needs the ADR-0014 migration dance; the QA
  graph does not change dim discipline.
