"""FastAPI entrypoint. OpenAI-compatible shape; ingest is async signed-URL."""

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_lab.adapters.db import db_ping, get_engine
from rag_lab.adapters.documents import DocumentStorePort, PostgresDocumentStore
from rag_lab.adapters.embeddings import EmbeddingPort, OpenAICompatibleEmbedding
from rag_lab.adapters.llm_openai import LlmPort, OpenAICompatibleLlm
from rag_lab.adapters.retrieval import PostgresRetrieval, RetrievalPort
from rag_lab.adapters.storage import (
    AzuriteBlobStorage,
    BlobStat,
    InMemoryStorage,
    StoragePort,
    blob_name,
)
from rag_lab.adapters.traces import PostgresTraceStore, TraceStorePort
from rag_lab.api.ui import register_ui
from rag_lab.core.config import Settings
from rag_lab.core.models import (
    Citation,
    Document,
    DocumentStatusView,
    IngestErrorCode,
    QaTrace,
    Trace,
)
from rag_lab.graph.qa import QaGenerationError, run_qa

logger = logging.getLogger(__name__)
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
ALLOWED_CONTENT_TYPES = {"text/plain", "text/markdown", "application/pdf"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    citations: list[Citation] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    db: bool
    blob: bool


class ModelEntry(BaseModel):
    id: str
    object: str


class ModelsResponse(BaseModel):
    object: str
    data: list[ModelEntry]


class InitiateUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(gt=0)


class InitiateUploadResponse(BaseModel):
    document_id: str
    upload_url: str
    expires_in: int


def _build_storage(settings: Settings, stored: StoragePort | None) -> StoragePort:
    if stored is not None:
        return stored
    if settings.azurite_connection_string:
        return AzuriteBlobStorage(settings.azurite_connection_string, settings.blob_container)
    return InMemoryStorage()


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


def _mark_failed_or_404(
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
        return _mark_failed_or_404(doc_store, document_id, "upload_incomplete")
    code = _verify_blob(meta, doc.mime, max_upload_bytes)
    if code is not None:
        return _mark_failed_or_404(doc_store, document_id, code)
    uploaded = doc_store.mark_uploaded(document_id)
    if uploaded is None:
        raise HTTPException(status_code=404, detail="unknown document")
    return uploaded


def _default_doc_store(resolved: Settings, stored: DocumentStorePort | None) -> DocumentStorePort:
    if stored is not None:
        return stored
    return PostgresDocumentStore(get_engine(resolved.database_url))


def _default_embedder(resolved: Settings, stored: EmbeddingPort | None) -> EmbeddingPort:
    if stored is not None:
        return stored
    return OpenAICompatibleEmbedding(
        base_url=resolved.embedding_base_url,
        api_key=resolved.embedding_api_key,
        model=resolved.embedding_model,
    )


def _default_retriever(resolved: Settings, stored: RetrievalPort | None) -> RetrievalPort:
    if stored is not None:
        return stored
    return PostgresRetrieval(
        get_engine(resolved.database_url),
        expected_dim=resolved.embedding_dimensions,
    )


def _default_llm(resolved: Settings, stored: LlmPort | None) -> LlmPort:
    if stored is not None:
        return stored
    return OpenAICompatibleLlm(
        base_url=resolved.openai_base_url,
        api_key=resolved.openai_api_key,
        model=resolved.openai_model,
    )


def _default_trace_store(resolved: Settings, stored: TraceStorePort | None) -> TraceStorePort:
    if stored is not None:
        return stored
    return PostgresTraceStore(get_engine(resolved.database_url))


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
    """Run the QA graph → save trace fail-open."""
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
    candidates = outcome["candidates"]
    reranked = outcome["reranked"]
    prompt = outcome["prompt"]
    answer = outcome["answer"]
    trace = Trace(
        trace_id=trace_id,
        query=query,
        retrieved=candidates,
        rerank_order=[result.chunk.id or "" for result in reranked],
        prompt=prompt,
    )
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
    citations = [
        Citation(
            chunk_id=result.chunk.id or "",
            source=result.chunk.title,
            title=result.chunk.title,
            heading_path=result.chunk.heading_path,
            page=result.chunk.page,
        )
        for result in reranked
    ]
    try:
        trace_store.save(trace, answer, citations)
    except Exception:
        logger.exception("trace save failed for %s", trace_id)
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


def _storage_of(app: FastAPI, resolved: Settings) -> StoragePort:
    return _build_storage(resolved, cast(StoragePort | None, app.state.storage))


def _ensure_emulator_cors(store: StoragePort, origins: list[str]) -> None:
    """Write dev CORS rules at startup; fail open so the API still serves."""
    try:
        store.ensure_cors(origins)
    except Exception:
        logger.exception("blob CORS ensure failed")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    resolved = cast(Settings, app.state.settings)
    _ensure_emulator_cors(_storage_of(app, resolved), resolved.blob_cors_origins)
    yield


def _doc_store_of(app: FastAPI, resolved: Settings) -> DocumentStorePort:
    return _default_doc_store(resolved, cast(DocumentStorePort | None, app.state.doc_store))


def _embedder_of(app: FastAPI, resolved: Settings) -> EmbeddingPort:
    return _default_embedder(resolved, cast(EmbeddingPort | None, app.state.embedder))


def _retriever_of(app: FastAPI, resolved: Settings) -> RetrievalPort:
    return _default_retriever(resolved, cast(RetrievalPort | None, app.state.retriever))


def _llm_of(app: FastAPI, resolved: Settings) -> LlmPort:
    return _default_llm(resolved, cast(LlmPort | None, app.state.llm))


def _trace_store_of(app: FastAPI, resolved: Settings) -> TraceStorePort:
    return _default_trace_store(resolved, cast(TraceStorePort | None, app.state.trace_store))


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
        store = _storage_of(app, resolved)
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
            embedder=_embedder_of(app, resolved),
            retriever=_retriever_of(app, resolved),
            llm=_llm_of(app, resolved),
            trace_store=_trace_store_of(app, resolved),
            retrieval_candidates=resolved.retrieval_candidates,
            answer_context_chunks=resolved.answer_context_chunks,
        )

    @app.get("/v1/traces/{trace_id}")
    def get_trace(trace_id: str) -> QaTrace:
        return _get_trace_record(_trace_store_of(app, resolved), trace_id)

    @app.post("/v1/kb/uploads/initiate")
    def initiate_upload(body: InitiateUploadRequest) -> InitiateUploadResponse:
        return _initiate_upload(
            body,
            max_upload_bytes=resolved.max_upload_bytes,
            storage=_storage_of(app, resolved),
            doc_store=_doc_store_of(app, resolved),
        )

    @app.post("/v1/kb/uploads/{document_id}/complete")
    def complete_upload(document_id: str) -> DocumentStatusView:
        return _complete_upload(
            document_id,
            max_upload_bytes=resolved.max_upload_bytes,
            storage=_storage_of(app, resolved),
            doc_store=_doc_store_of(app, resolved),
        )

    @app.get("/v1/kb/documents/{document_id}")
    def get_document(document_id: str) -> DocumentStatusView:
        view = _doc_store_of(app, resolved).get_status(document_id)
        if view is None:
            raise HTTPException(status_code=404, detail="unknown document")
        return view

    return app


app = create_app()
