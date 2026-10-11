"""FastAPI entrypoint. OpenAI-compatible shape; ingest is async signed-URL."""

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException

from rag_lab.adapters.db import db_ping
from rag_lab.adapters.documents import DocumentStorePort
from rag_lab.adapters.embeddings import EmbeddingPort
from rag_lab.adapters.llm_openai import LlmPort
from rag_lab.adapters.retrieval import RetrievalPort
from rag_lab.adapters.storage import BlobStat, StoragePort, blob_name
from rag_lab.adapters.traces import TraceStorePort
from rag_lab.api.deps import (
    ensure_emulator_cors,
    resolve_doc_store,
    resolve_embedder,
    resolve_llm,
    resolve_retriever,
    resolve_storage,
    resolve_trace_store,
)
from rag_lab.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    HealthResponse,
    InitiateUploadRequest,
    InitiateUploadResponse,
    ModelEntry,
    ModelsResponse,
)
from rag_lab.api.ui import register_ui
from rag_lab.core.config import Settings
from rag_lab.core.models import (
    Citation,
    Document,
    DocumentStatusView,
    IngestErrorCode,
    QaTrace,
    RetrievalResult,
    Trace,
)
from rag_lab.graph.qa import QaGenerationError, QaOutcome, run_qa

logger = logging.getLogger(__name__)
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
ALLOWED_CONTENT_TYPES = {"text/plain", "text/markdown", "application/pdf"}


def _verify_blob(meta: BlobStat, declared_mime: str, max_bytes: int) -> IngestErrorCode | None:
    """Check HEAD results against the declared upload. None means acceptable."""
    if meta.size > max_bytes:
        return "size_exceeded"
    if meta.size == 0:
        return "empty_content"
    if meta.content_type != declared_mime:
        return "type_mismatch"
    return None


def _initiate_upload(
    body: InitiateUploadRequest,
    *,
    max_upload_bytes: int,
    storage: StoragePort,
    doc_store: DocumentStorePort,
) -> InitiateUploadResponse:
    """Validate → persist initiated row → mint signed URL.

    The row is persisted before any signed URL is minted so a DB failure
    never hands out a URL no document references. If URL minting fails
    afterwards, the row is failed best-effort (upload_incomplete) rather
    than left initiated forever.
    """
    if body.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="unsupported media type")
    if body.size_bytes > max_upload_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    sanitized = SAFE_NAME.sub("_", body.filename.rsplit("/", 1)[-1])
    if not sanitized.strip("._"):
        raise HTTPException(status_code=422, detail="invalid filename")
    key = f"{uuid.uuid4()}-{sanitized}"
    canonical = storage.blob_url(key)
    document = Document(
        source=body.filename,
        title=sanitized,
        mime=body.content_type,
        blob_url=canonical,
    )
    try:
        document_id = doc_store.initiate(document)
    except Exception:
        logger.exception("document initiate failed")
        raise HTTPException(status_code=500, detail="ingest failed") from None
    try:
        upload_url, expires_in = storage.issue_upload_url(key, body.content_type)
    except Exception:
        logger.exception("upload URL issue failed for %s", document_id)
        try:
            doc_store.fail(document_id, "upload_incomplete")
        except Exception:
            logger.exception("upload URL cleanup failed for %s", document_id)
        raise HTTPException(status_code=500, detail="ingest failed") from None
    return InitiateUploadResponse(
        document_id=document_id, upload_url=upload_url, expires_in=expires_in
    )


def fail_document_or_404(
    doc_store: DocumentStorePort, document_id: str, error_code: IngestErrorCode
) -> DocumentStatusView:
    """Record a typed ingest failure, or 404 when the document is unknown."""
    failed = doc_store.fail(document_id, error_code)
    if failed is None:
        raise HTTPException(status_code=404, detail="unknown document")
    return failed


def _last_user_message(messages: list[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content
    return None


def _complete_upload(
    document_id: str,
    *,
    max_upload_bytes: int,
    storage: StoragePort,
    doc_store: DocumentStorePort,
) -> DocumentStatusView:
    """HEAD the blob → flip initiated to uploaded, or record a typed failure.

    Contract note: validation-time faults on initiate surface as 4xx, while
    completion-time faults surface as 200 with a failed status view carrying
    a typed error_code — the client must check the body, not just the code.
    """
    current = doc_store.get_status(document_id)
    if current is None:
        raise HTTPException(status_code=404, detail="unknown document")
    if current.status != "initiated":
        return current
    doc = doc_store.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="unknown document")
    try:
        meta = storage.stat(blob_name(doc.blob_url))
    except Exception:
        logger.exception("blob stat failed for %s", document_id)
        raise HTTPException(status_code=500, detail="ingest failed") from None
    if meta is None:
        return fail_document_or_404(doc_store, document_id, "upload_incomplete")
    code = _verify_blob(meta, doc.mime, max_upload_bytes)
    if code is not None:
        return fail_document_or_404(doc_store, document_id, code)
    uploaded = doc_store.mark_uploaded(document_id)
    if uploaded is None:
        raise HTTPException(status_code=404, detail="unknown document")
    return uploaded


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


def _answer_question(
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


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    resolved = cast(Settings, app.state.settings)
    ensure_emulator_cors(resolve_storage(app, resolved), resolved.blob_cors_origins)
    yield


def _get_trace_record(trace_store: TraceStorePort, trace_id: str) -> QaTrace:
    record = trace_store.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown trace")
    return record


def create_app(
    settings: Settings | None = None,
    storage: StoragePort | None = None,
    doc_store: DocumentStorePort | None = None,
    embedder: EmbeddingPort | None = None,
    retriever: RetrievalPort | None = None,
    llm: LlmPort | None = None,
    trace_store: TraceStorePort | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    app = FastAPI(title="rag-lab", version="0.1.0", lifespan=_lifespan)
    app.state.settings = resolved
    app.state.storage = storage
    app.state.doc_store = doc_store
    app.state.embedder = embedder
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.trace_store = trace_store
    register_ui(app)

    @app.get("/health")
    def health() -> HealthResponse:
        store = resolve_storage(app, resolved)
        return HealthResponse(
            status="ok",
            db=db_ping(resolved.database_url),
            blob=store.ping(),
        )

    @app.get("/v1/models")
    def list_models() -> ModelsResponse:
        return ModelsResponse(
            object="list",
            data=[
                ModelEntry(id=resolved.openai_model, object="model"),
                ModelEntry(id=resolved.embedding_model, object="model"),
            ],
        )

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest) -> ChatCompletionResponse:
        query = _last_user_message(body.messages)
        if query is None:
            raise HTTPException(status_code=422, detail="no user message")
        return _answer_question(
            query,
            body.model,
            embedder=resolve_embedder(app, resolved),
            retriever=resolve_retriever(app, resolved),
            llm=resolve_llm(app, resolved),
            trace_store=resolve_trace_store(app, resolved),
            retrieval_candidates=resolved.retrieval_candidates,
            answer_context_chunks=resolved.answer_context_chunks,
        )

    @app.get("/v1/traces/{trace_id}")
    def get_trace(trace_id: str) -> QaTrace:
        return _get_trace_record(resolve_trace_store(app, resolved), trace_id)

    @app.post("/v1/kb/uploads/initiate")
    def initiate_upload(body: InitiateUploadRequest) -> InitiateUploadResponse:
        return _initiate_upload(
            body,
            max_upload_bytes=resolved.max_upload_bytes,
            storage=resolve_storage(app, resolved),
            doc_store=resolve_doc_store(app, resolved),
        )

    @app.post("/v1/kb/uploads/{document_id}/complete")
    def complete_upload(document_id: str) -> DocumentStatusView:
        return _complete_upload(
            document_id,
            max_upload_bytes=resolved.max_upload_bytes,
            storage=resolve_storage(app, resolved),
            doc_store=resolve_doc_store(app, resolved),
        )

    @app.get("/v1/kb/documents/{document_id}")
    def get_document(document_id: str) -> DocumentStatusView:
        view = resolve_doc_store(app, resolved).get_status(document_id)
        if view is None:
            raise HTTPException(status_code=404, detail="unknown document")
        return view

    return app


app = create_app()
