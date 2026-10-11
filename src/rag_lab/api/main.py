"""FastAPI entrypoint. OpenAI-compatible shape; RAG wiring lands in Phase 3."""

import logging
import re
import uuid
from typing import Annotated, cast

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_lab.adapters.db import db_ping, get_engine
from rag_lab.adapters.documents import DocumentStorePort, PostgresDocumentStore
from rag_lab.adapters.storage import (
    AzuriteBlobStorage,
    InMemoryStorage,
    StoragePort,
)
from rag_lab.core.config import Settings
from rag_lab.core.models import Chunk, Document
from rag_lab.ingest.loaders import LoaderPort, TextLoader

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


class UploadResponse(BaseModel):
    name: str
    blob_url: str
    bytes: int
    document_id: str
    chunk_count: int


def _build_storage(settings: Settings, stored: StoragePort | None) -> StoragePort:
    if stored is not None:
        return stored
    if settings.azurite_connection_string:
        return AzuriteBlobStorage(settings.azurite_connection_string, settings.blob_container)
    return InMemoryStorage()


def _select_loader(
    loaders: list[LoaderPort], filename: str, content_type: str
) -> LoaderPort | None:
    return next(
        (candidate for candidate in loaders if candidate.can_handle(filename, content_type)),
        None,
    )


def _load_chunks(loader: LoaderPort, data: bytes, source: str) -> list[Chunk] | None:
    """Decode bytes via loader. None means unprocessable (caller maps to 422)."""
    try:
        return loader.load(data, source=source) or None
    except (UnicodeDecodeError, ValueError):
        return None


async def _upload_file(
    file: UploadFile,
    *,
    max_upload_bytes: int,
    loaders: list[LoaderPort],
    storage: StoragePort,
    doc_store: DocumentStorePort,
) -> UploadResponse:
    """Validate → load → put blob → persist document+chunks. One side effect per line."""
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="unsupported media type")
    data = await file.read()
    if len(data) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    if not data:
        raise HTTPException(status_code=422, detail="empty file")
    original = file.filename or "upload"
    loader = _select_loader(loaders, original, content_type)
    if loader is None:
        # Honest until the PDF slice lands a loader: allowlisted but not ingestible yet.
        raise HTTPException(status_code=415, detail="unsupported media type")
    loaded = _load_chunks(loader, data, source=original)
    if loaded is None:
        raise HTTPException(status_code=422, detail="unprocessable file")
    filename = SAFE_NAME.sub("_", original)
    name = f"{uuid.uuid4()}-{filename}"
    url = storage.put(name, data, content_type)
    document = Document(
        source=original,
        title=loaded[0].title or filename,
        mime=content_type,
        blob_url=url,
    )
    try:
        document_id = doc_store.save(document, loaded)
    except Exception as e:
        logger.exception("document persist failed")
        raise HTTPException(status_code=500, detail="ingest failed") from e
    return UploadResponse(
        name=name,
        blob_url=url,
        bytes=len(data),
        document_id=document_id,
        chunk_count=len(loaded),
    )


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
    app.state.loaders = [TextLoader()]

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

    @app.post("/v1/kb/upload")
    async def kb_upload(file: Annotated[UploadFile, File()]) -> UploadResponse:
        return await _upload_file(
            file,
            max_upload_bytes=resolved.max_upload_bytes,
            loaders=cast("list[LoaderPort]", app.state.loaders),
            storage=get_storage(),
            doc_store=get_doc_store(),
        )

    return app


app = create_app()
