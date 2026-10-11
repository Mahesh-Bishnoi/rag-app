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
from rag_lab.adapters.embeddings import EmbeddingPort, OpenAICompatibleEmbedding
from rag_lab.adapters.storage import (
    AzuriteBlobStorage,
    InMemoryStorage,
    StoragePort,
)
from rag_lab.core.config import Settings
from rag_lab.core.models import Chunk, Document
from rag_lab.ingest.chunking import split_chunks
from rag_lab.ingest.loaders import LoaderPort, PdfLoader, TextLoader

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


def _embed_chunks(chunks: list[Chunk], embedder: EmbeddingPort) -> list[Chunk] | None:
    """Attach one vector per chunk. None means embed failed (caller maps to 500)."""
    try:
        vectors = embedder.embed([chunk.content for chunk in chunks])
    except Exception:
        return None
    if len(vectors) != len(chunks):
        return None
    return [
        chunk.model_copy(update={"embedding": vector})
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


async def _upload_file(
    file: UploadFile,
    *,
    max_upload_bytes: int,
    loaders: list[LoaderPort],
    storage: StoragePort,
    doc_store: DocumentStorePort,
    embedder: EmbeddingPort | None = None,
) -> UploadResponse:
    """Validate → load → split → embed → put blob → persist. One side effect per line."""
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
        raise HTTPException(status_code=415, detail="unsupported media type")
    loaded = _load_chunks(loader, data, source=original)
    if loaded is None:
        raise HTTPException(status_code=422, detail="unprocessable file")
    chunks = split_chunks(loaded)
    if not chunks:
        raise HTTPException(status_code=422, detail="unprocessable file")
    if embedder is not None:
        embedded = _embed_chunks(chunks, embedder)
        if embedded is None:
            logger.exception("chunk embedding failed")
            raise HTTPException(status_code=500, detail="ingest failed")
        chunks = embedded
    filename = SAFE_NAME.sub("_", original)
    name = f"{uuid.uuid4()}-{filename}"
    url = storage.put(name, data, content_type)
    document = Document(
        source=original,
        title=chunks[0].title or filename,
        mime=content_type,
        blob_url=url,
    )
    try:
        document_id = doc_store.save(document, chunks)
    except Exception as e:
        logger.exception("document persist failed")
        raise HTTPException(status_code=500, detail="ingest failed") from e
    return UploadResponse(
        name=name,
        blob_url=url,
        bytes=len(data),
        document_id=document_id,
        chunk_count=len(chunks),
    )


def create_app(
    settings: Settings | None = None,
    storage: StoragePort | None = None,
    doc_store: DocumentStorePort | None = None,
    embedder: EmbeddingPort | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    app = FastAPI(title="rag-lab", version="0.1.0")
    app.state.settings = resolved
    app.state.storage = storage
    app.state.doc_store = doc_store or PostgresDocumentStore(get_engine(resolved.database_url))
    app.state.loaders = [TextLoader(), PdfLoader()]
    app.state.embedder = embedder

    def get_storage() -> StoragePort:
        stored: StoragePort | None = cast("StoragePort | None", app.state.storage)
        return _build_storage(resolved, stored)

    def get_doc_store() -> DocumentStorePort:
        return cast("DocumentStorePort", app.state.doc_store)

    def get_embedder() -> EmbeddingPort | None:
        return cast("EmbeddingPort | None", app.state.embedder)

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
            embedder=get_embedder(),
        )

    return app


def _default_embedder(resolved: Settings) -> EmbeddingPort:
    return OpenAICompatibleEmbedding(
        base_url=resolved.embedding_base_url,
        api_key=resolved.embedding_api_key,
        model=resolved.embedding_model,
    )


_module_settings = Settings()
app = create_app(settings=_module_settings, embedder=_default_embedder(_module_settings))
