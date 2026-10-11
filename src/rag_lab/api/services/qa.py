"""QA service: run the graph → log → save trace fail-open → respond.

Routers handle HTTP; this module owns the question-answering workflow.
Builders are pure and stay private; answer_question and last_user_message
are the package surface the chat route uses.
"""

import logging
import time
import uuid

from fastapi import HTTPException

from rag_lab.adapters.embeddings import EmbeddingPort
from rag_lab.adapters.llm_openai import LlmPort
from rag_lab.adapters.retrieval import RetrievalPort
from rag_lab.adapters.traces import TraceStorePort
from rag_lab.api.schemas import ChatCompletionChoice, ChatCompletionResponse, ChatMessage
from rag_lab.core.models import Citation, RetrievalResult, Trace
from rag_lab.graph.qa import QaGenerationError, QaOutcome, run_qa

logger = logging.getLogger(__name__)


def last_user_message(messages: list[ChatMessage]) -> str | None:
    """Most recent non-blank user message, or None when there isn't one."""
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content
    return None


def _build_trace(trace_id: str, query: str, outcome: QaOutcome) -> Trace:
    """Trace record from graph outcome. Pure."""
    return Trace(
        trace_id=trace_id,
        query=query,
        retrieved=outcome["candidates"],
        rerank_order=[result.chunk.id or "" for result in outcome["reranked"]],
        prompt=outcome["prompt"],
    )


def _trace_citations(reranked: list[RetrievalResult]) -> list[Citation]:
    """Citations from reranked chunks. Pure."""
    return [
        Citation(
            chunk_id=result.chunk.id or "",
            source=result.chunk.title,
            title=result.chunk.title,
            heading_path=result.chunk.heading_path,
            page=result.chunk.page,
        )
        for result in reranked
    ]


def _build_chat_response(
    trace_id: str, model: str, answer: str, citations: list[Citation]
) -> ChatCompletionResponse:
    """HTTP response envelope for a finished answer. Pure."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{trace_id}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        citations=citations,
    )


def _log_answered(trace: Trace) -> None:
    logger.info(
        "qa answered",
        extra={
            "event": "qa_answered",
            "trace_id": trace.trace_id,
            "query_chars": len(trace.query),
            "retrieved": [
                (result.chunk.id, result.vector_score, result.bm25_score, result.fused_rank)
                for result in trace.retrieved
            ],
            "rerank_order": trace.rerank_order,
            "prompt_chars": len(trace.prompt),
        },
    )


def answer_question(
    query: str,
    model: str,
    *,
    embedder: EmbeddingPort,
    retriever: RetrievalPort,
    llm: LlmPort,
    trace_store: TraceStorePort,
    retrieval_candidates: int,
    answer_context_chunks: int,
) -> ChatCompletionResponse:
    """Run the QA graph → log → save trace fail-open → respond."""
    trace_id = uuid.uuid4().hex
    try:
        outcome = run_qa(
            embedder=embedder,
            retriever=retriever,
            llm=llm,
            query=query,
            retrieval_candidates=retrieval_candidates,
            answer_context_chunks=answer_context_chunks,
        )
    except QaGenerationError:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail="generation failed") from None
    except Exception:
        logger.exception("query retrieval failed")
        raise HTTPException(status_code=500, detail="retrieval failed") from None
    trace = _build_trace(trace_id, query, outcome)
    _log_answered(trace)
    citations = _trace_citations(outcome["reranked"])
    try:
        trace_store.save(trace, outcome["answer"], citations)
    except Exception:
        logger.exception("trace save failed for %s", trace_id)
    return _build_chat_response(trace_id, model, outcome["answer"], citations)
