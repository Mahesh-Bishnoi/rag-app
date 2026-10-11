"""Chat route: OpenAI-compatible completions. Thin handler over the QA service."""

from fastapi import APIRouter, FastAPI, HTTPException

from rag_lab.api.deps import (
    resolve_embedder,
    resolve_llm,
    resolve_retriever,
    resolve_trace_store,
)
from rag_lab.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from rag_lab.api.services.qa import answer_question, last_user_message
from rag_lab.core.config import Settings


def create_chat_router(app: FastAPI, resolved: Settings) -> APIRouter:
    router = APIRouter(tags=["chat"])

    @router.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest) -> ChatCompletionResponse:
        query = last_user_message(body.messages)
        if query is None:
            raise HTTPException(status_code=422, detail="no user message")
        return answer_question(
            query,
            body.model,
            embedder=resolve_embedder(app, resolved),
            retriever=resolve_retriever(app, resolved),
            llm=resolve_llm(app, resolved),
            trace_store=resolve_trace_store(app, resolved),
            retrieval_candidates=resolved.retrieval_candidates,
            answer_context_chunks=resolved.answer_context_chunks,
        )

    return router
