"""FastAPI entrypoint. OpenAI-compatible shape; RAG wiring lands in Phase 3."""

import re
import uuid
from typing import Any, cast

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_lab.adapters.db import db_ping
from rag_lab.adapters.storage import (
    AzuriteBlobStorage,
    InMemoryStorage,
    StoragePort,
)
from rag_lab.core.config import Settings
from rag_lab.ingest.loaders import TextLoader

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
ALLOWED_CONTENT_TYPES = {"text/plain", "text/markdown", "application/pdf"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


def _build_storage(settings: Settings, stored: StoragePort | None) -> StoragePort:
    if stored is not None:
        return stored
    if settings.azurite_connection_string:
        return AzuriteBlobStorage(settings.azurite_connection_string, settings.blob_container)
    return InMemoryStorage()


def create_app(settings: Settings | None = None, storage: StoragePort | None = None) -> FastAPI:
    resolved = settings or Settings()
    app = FastAPI(title="rag-lab", version="0.1.0")
    app.state.settings = resolved
    app.state.storage = storage
    app.state.loaders = [TextLoader()]

    def get_storage() -> StoragePort:
        stored: StoragePort | None = cast("StoragePort | None", app.state.storage)
        return _build_storage(resolved, stored)

    @app.get("/health")
    def health() -> dict[str, Any]:
        store = get_storage()
        return {
            "status": "ok",
            "db": db_ping(resolved.database_url),
            "blob": store.ping(),
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": resolved.openai_model, "object": "model"},
                {"id": resolved.embedding_model, "object": "model"},
            ],
        }

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
    async def kb_upload(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        content_type = file.content_type or "application/octet-stream"
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="unsupported media type")
        data = await file.read()
        if len(data) > resolved.max_upload_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        if not data:
            raise HTTPException(status_code=422, detail="empty file")
        filename = SAFE_NAME.sub("_", file.filename or "upload")
        name = f"{uuid.uuid4()}-{filename}"
        url = get_storage().put(name, data, content_type)
        return {"name": name, "blob_url": url, "bytes": len(data)}

    return app


app = create_app()
