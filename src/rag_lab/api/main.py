"""FastAPI entrypoint. OpenAI-compatible shape; ingest is async signed-URL.

Thin composition root: state + lifespan + router registration live here.
Chat answers live in api/services/qa.py, uploads in api/services/uploads.py,
endpoints in api/routes/*. Nothing else belongs in this file.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI

from rag_lab.adapters.documents import DocumentStorePort
from rag_lab.adapters.embeddings import EmbeddingPort
from rag_lab.adapters.llm_openai import LlmPort
from rag_lab.adapters.retrieval import RetrievalPort
from rag_lab.adapters.storage import StoragePort
from rag_lab.adapters.traces import TraceStorePort
from rag_lab.api.deps import ensure_emulator_cors, resolve_storage
from rag_lab.api.routes.chat import create_chat_router
from rag_lab.api.routes.health import create_health_router
from rag_lab.api.routes.kb import create_kb_router
from rag_lab.api.routes.traces import create_traces_router
from rag_lab.api.ui import register_ui
from rag_lab.core.config import Settings


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    resolved = cast(Settings, app.state.settings)
    ensure_emulator_cors(resolve_storage(app, resolved), resolved.blob_cors_origins)
    yield


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
    app.include_router(create_health_router(app, resolved))
    app.include_router(create_chat_router(app, resolved))
    app.include_router(create_traces_router(app, resolved))
    app.include_router(create_kb_router(app, resolved))
    return app


app = create_app()
