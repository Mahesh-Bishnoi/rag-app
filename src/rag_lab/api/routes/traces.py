"""Trace route: read back a persisted QA trace. Thin handler."""

from fastapi import APIRouter, FastAPI, HTTPException

from rag_lab.adapters.traces import TraceStorePort
from rag_lab.api.deps import resolve_trace_store
from rag_lab.core.config import Settings
from rag_lab.core.models import QaTrace


def _get_trace_record(trace_store: TraceStorePort, trace_id: str) -> QaTrace:
    record = trace_store.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown trace")
    return record


def create_traces_router(app: FastAPI, resolved: Settings) -> APIRouter:
    router = APIRouter(tags=["traces"])

    @router.get("/v1/traces/{trace_id}")
    def get_trace(trace_id: str) -> QaTrace:
        return _get_trace_record(resolve_trace_store(app, resolved), trace_id)

    return router
