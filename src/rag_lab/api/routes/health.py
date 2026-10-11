"""System routes: liveness probe and model listing. No business logic."""

from fastapi import APIRouter, FastAPI

from rag_lab.adapters.db import db_ping
from rag_lab.api.deps import resolve_storage
from rag_lab.api.schemas import HealthResponse, ModelEntry, ModelsResponse
from rag_lab.core.config import Settings


def create_health_router(app: FastAPI, resolved: Settings) -> APIRouter:
    router = APIRouter(tags=["system"])

    @router.get("/health")
    def health() -> HealthResponse:
        store = resolve_storage(app, resolved)
        return HealthResponse(
            status="ok",
            db=db_ping(resolved.database_url),
            blob=store.ping(),
        )

    @router.get("/v1/models")
    def list_models() -> ModelsResponse:
        return ModelsResponse(
            object="list",
            data=[
                ModelEntry(id=resolved.openai_model, object="model"),
                ModelEntry(id=resolved.embedding_model, object="model"),
            ],
        )

    return router
