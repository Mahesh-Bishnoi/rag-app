"""KB routes: signed-URL upload initiate/complete plus document status reads."""

from fastapi import APIRouter, FastAPI, HTTPException

from rag_lab.api.deps import resolve_doc_store, resolve_storage
from rag_lab.api.schemas import InitiateUploadRequest, InitiateUploadResponse
from rag_lab.api.services.uploads import complete_upload, initiate_upload
from rag_lab.core.config import Settings
from rag_lab.core.models import DocumentStatusView


def create_kb_router(app: FastAPI, resolved: Settings) -> APIRouter:
    router = APIRouter(prefix="/v1/kb", tags=["kb"])

    @router.post("/uploads/initiate")
    def initiate(body: InitiateUploadRequest) -> InitiateUploadResponse:
        return initiate_upload(
            body,
            max_upload_bytes=resolved.max_upload_bytes,
            storage=resolve_storage(app, resolved),
            doc_store=resolve_doc_store(app, resolved),
        )

    @router.post("/uploads/{document_id}/complete")
    def complete(document_id: str) -> DocumentStatusView:
        return complete_upload(
            document_id,
            max_upload_bytes=resolved.max_upload_bytes,
            storage=resolve_storage(app, resolved),
            doc_store=resolve_doc_store(app, resolved),
        )

    @router.get("/documents/{document_id}")
    def get_document(document_id: str) -> DocumentStatusView:
        view = resolve_doc_store(app, resolved).get_status(document_id)
        if view is None:
            raise HTTPException(status_code=404, detail="unknown document")
        return view

    return router
