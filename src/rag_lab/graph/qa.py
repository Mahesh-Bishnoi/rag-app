"""QA graph: LangGraph owns the embed → retrieve → rerank → generate flow.

Nodes are thin wrappers over the injected ports; the graph wires the order.
Failures propagate to the caller (the API maps them to 500s) — nodes never
swallow. The prompt is part of the state so traces stay observable.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_lab.adapters.embeddings import EmbeddingPort
from rag_lab.adapters.generation import complete, grounded_prompt
from rag_lab.adapters.llm_openai import LlmPort
from rag_lab.adapters.rerank import rerank_candidates
from rag_lab.adapters.retrieval import RetrievalPort
from rag_lab.core.models import RetrievalResult


class QaState(TypedDict, total=False):
    query: str
    query_embedding: list[float]
    candidates: list[RetrievalResult]
    reranked: list[RetrievalResult]
    prompt: str
    answer: str


class QaOutcome(TypedDict):
    """What the API needs after the graph runs: trace fields + answer."""

    candidates: list[RetrievalResult]
    reranked: list[RetrievalResult]
    prompt: str
    answer: str


class QaGenerationError(Exception):
    """The generate node failed (LLM transport or typed-parse error)."""


def build_qa_graph(
    embedder: EmbeddingPort,
    retriever: RetrievalPort,
    llm: LlmPort,
    retrieval_candidates: int,
    answer_context_chunks: int,
) -> CompiledStateGraph[QaState, None, QaState, QaState]:
    """Wire the four QA stages in order. Ports are injected, never constructed."""

    def _embed(state: QaState) -> dict[str, object]:
        vectors = embedder.embed([state["query"]])
        if len(vectors) != 1:
            raise ValueError(f"expected 1 query vector, got {len(vectors)}")
        return {"query_embedding": vectors[0]}

    def _retrieve(state: QaState) -> dict[str, object]:
        return {
            "candidates": retriever.search(
                state["query_embedding"], state["query"], retrieval_candidates
            )
        }

    def _rerank(state: QaState) -> dict[str, object]:
        return {
            "reranked": rerank_candidates(
                state["query"], state["candidates"], answer_context_chunks
            )
        }

    def _generate(state: QaState) -> dict[str, object]:
        prompt = grounded_prompt(
            state["query"], [result.chunk.content for result in state["reranked"]]
        )
        try:
            answer = complete(llm, prompt).answer
        except Exception as e:
            raise QaGenerationError(f"generate failed for query {state['query']!r}") from e
        return {"prompt": prompt, "answer": answer}

    graph = StateGraph(QaState)
    graph.add_node("embed", _embed)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("rerank", _rerank)
    graph.add_node("generate", _generate)
    graph.set_entry_point("embed")
    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_qa(
    *,
    embedder: EmbeddingPort,
    retriever: RetrievalPort,
    llm: LlmPort,
    query: str,
    retrieval_candidates: int,
    answer_context_chunks: int,
) -> QaOutcome:
    """Run one question through the QA graph. Raises on embed/retrieve/LLM failure."""
    raw = build_qa_graph(
        embedder, retriever, llm, retrieval_candidates, answer_context_chunks
    ).invoke({"query": query})
    return QaOutcome(
        candidates=raw["candidates"],
        reranked=raw["reranked"],
        prompt=raw["prompt"],
        answer=raw["answer"],
    )
