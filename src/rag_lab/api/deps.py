"""Port resolution for HTTP handlers. Test double when injected, prod default otherwise."""

import logging
from collections.abc import Callable
from typing import TypeVar, cast

from fastapi import FastAPI

from rag_lab.adapters.db import get_engine
from rag_lab.adapters.documents import DocumentStorePort, PostgresDocumentStore
from rag_lab.adapters.embeddings import EmbeddingPort, OpenAICompatibleEmbedding
from rag_lab.adapters.llm_openai import LlmPort, OpenAICompatibleLlm
from rag_lab.adapters.retrieval import PostgresRetrieval, RetrievalPort
from rag_lab.adapters.storage import (
    AzuriteBlobStorage,
    InMemoryStorage,
    StoragePort,
)
from rag_lab.adapters.traces import PostgresTraceStore, TraceStorePort
from rag_lab.core.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def resolve_or_create(stored: T | None, factory: Callable[[], T]) -> T:
    """Return the injected test double, or build the prod default."""
    return stored if stored is not None else factory()


def resolve_storage(app: FastAPI, resolved: Settings) -> StoragePort:
    def _default() -> StoragePort:
        if resolved.azurite_connection_string:
            return AzuriteBlobStorage(resolved.azurite_connection_string, resolved.blob_container)
        return InMemoryStorage()

    return resolve_or_create(cast(StoragePort | None, app.state.storage), _default)


def resolve_doc_store(app: FastAPI, resolved: Settings) -> DocumentStorePort:
    return resolve_or_create(
        cast(DocumentStorePort | None, app.state.doc_store),
        lambda: PostgresDocumentStore(get_engine(resolved.database_url)),
    )


def resolve_embedder(app: FastAPI, resolved: Settings) -> EmbeddingPort:
    return resolve_or_create(
        cast(EmbeddingPort | None, app.state.embedder),
        lambda: OpenAICompatibleEmbedding(
            base_url=resolved.embedding_base_url,
            api_key=resolved.embedding_api_key,
            model=resolved.embedding_model,
        ),
    )


def resolve_retriever(app: FastAPI, resolved: Settings) -> RetrievalPort:
    return resolve_or_create(
        cast(RetrievalPort | None, app.state.retriever),
        lambda: PostgresRetrieval(
            get_engine(resolved.database_url),
            expected_dim=resolved.embedding_dimensions,
        ),
    )


def resolve_llm(app: FastAPI, resolved: Settings) -> LlmPort:
    return resolve_or_create(
        cast(LlmPort | None, app.state.llm),
        lambda: OpenAICompatibleLlm(
            base_url=resolved.openai_base_url,
            api_key=resolved.openai_api_key,
            model=resolved.openai_model,
        ),
    )


def resolve_trace_store(app: FastAPI, resolved: Settings) -> TraceStorePort:
    return resolve_or_create(
        cast(TraceStorePort | None, app.state.trace_store),
        lambda: PostgresTraceStore(get_engine(resolved.database_url)),
    )


def ensure_emulator_cors(store: StoragePort, origins: list[str]) -> None:
    """Write dev CORS rules at startup; fail open so the API still serves."""
    try:
        store.ensure_cors(origins)
    except Exception:
        logger.exception("blob CORS ensure failed")
