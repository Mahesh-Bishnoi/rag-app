"""FastAPI entrypoint. OpenAI-compatible shape; ingest is async signed-URL."""

import logging
import re
import uuid
from typing import cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_lab.adapters.db import db_ping, get_engine
from rag_lab.adapters.documents import DocumentStorePort, PostgresDocumentStore
from rag_lab.adapters.storage import (
    AzuriteBlobStorage,
    BlobStat,
    InMemoryStorage,
    StoragePort,
)
from rag_lab.core.config import Settings
from rag_lab.core.models import Document, DocumentStatusView, IngestErrorCode
from rag_lab.worker.pipeline import blob_name

logger = logging.getLogger(__name__)
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
ALLOWED_CONTENT_TYPES = {"text/plain", "text/markdown", "application/pdf"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


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
    """Validate → mint signed URL → persist initiated row. One side effect per line."""
    if body.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="unsupported media type")
    if body.size_bytes > max_upload_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    sanitized = SAFE_NAME.sub("_", body.filename.rsplit("/", 1)[-1])
    if not sanitized.strip("._"):
        raise HTTPException(status_code=422, detail="invalid filename")
    key = f"{uuid.uuid4()}-{sanitized}"
    try:
        canonical = storage.blob_url(key)
        upload_url, expires_in = storage.issue_upload_url(key, body.content_type)
    except Exception:
        logger.exception("upload URL issue failed")
        raise HTTPException(status_code=500, detail="ingest failed") from None
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
    return InitiateUploadResponse(
        document_id=document_id, upload_url=upload_url, expires_in=expires_in
    )


def _fail_or_404(
    doc_store: DocumentStorePort, document_id: str, error_code: IngestErrorCode
) -> DocumentStatusView:
    failed = doc_store.fail(document_id, error_code)
    if failed is None:
        raise HTTPException(status_code=404, detail="unknown document")
    return failed


def _complete_upload(
    document_id: str,
    *,
    max_upload_bytes: int,
    storage: StoragePort,
    doc_store: DocumentStorePort,
) -> DocumentStatusView:
    """HEAD the blob → flip initiated to uploaded, or record a typed failure."""
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
        return _fail_or_404(doc_store, document_id, "upload_incomplete")
    code = _verify_blob(meta, doc.mime, max_upload_bytes)
    if code is not None:
        return _fail_or_404(doc_store, document_id, code)
    uploaded = doc_store.mark_uploaded(document_id)
    if uploaded is None:
        raise HTTPException(status_code=404, detail="unknown document")
    return uploaded


def create_app(
    settings: Settings | None = None,
    storage: StoragePort | None = None,
    doc_store: DocumentStorePort | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    app = FastAPI(title="rag-lab", version="0.1.0")
    app.state.settings = resolved
    app.state.storage = storage
    app.state.doc_store = doc_store or PostgresDocumentStore(get_engine(resolved.database_url))

    def get_storage() -> StoragePort:
        stored: StoragePort | None = cast("StoragePort | None", app.state.storage)
        return _build_storage(resolved, stored)

    def get_doc_store() -> DocumentStorePort:
        return cast("DocumentStorePort", app.state.doc_store)

    @app.get("/health")
    def health() -> HealthResponse:
        store = get_storage()
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
    def chat_completions(body: ChatCompletionRequest) -> JSONResponse:
        # Phase 3 wires retrieve → rerank → generate here.
        # Skeleton stays honest: validate shape, report not-wired.
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "rag pipeline not wired yet (Phase 3)",
                    "type": "not_implemented",
                }
            },
        )

    @app.post("/v1/kb/uploads/initiate")
    def initiate_upload(body: InitiateUploadRequest) -> InitiateUploadResponse:
        return _initiate_upload(
            body,
            max_upload_bytes=resolved.max_upload_bytes,
            storage=get_storage(),
            doc_store=get_doc_store(),
        )

    @app.post("/v1/kb/uploads/{document_id}/complete")
    def complete_upload(document_id: str) -> DocumentStatusView:
        return _complete_upload(
            document_id,
            max_upload_bytes=resolved.max_upload_bytes,
            storage=get_storage(),
            doc_store=get_doc_store(),
        )

    @app.get("/v1/kb/documents/{document_id}")
    def get_document(document_id: str) -> DocumentStatusView:
        view = get_doc_store().get_status(document_id)
        if view is None:
            raise HTTPException(status_code=404, detail="unknown document")
        return view

    return app


app = create_app()
