"""Upload service: validate → persist initiated row → mint signed URL / complete.

Routers handle HTTP; this module owns the KB upload workflow, including the
typed failure mapping both sides share. fail_document_or_404 stays public
for routers; the blob check stays private.
"""

import logging
import re
import uuid

from fastapi import HTTPException

from rag_lab.adapters.documents import DocumentStorePort
from rag_lab.adapters.storage import BlobStat, StoragePort, blob_name
from rag_lab.api.schemas import InitiateUploadRequest, InitiateUploadResponse
from rag_lab.core.models import Document, DocumentStatusView, IngestErrorCode

logger = logging.getLogger(__name__)
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
ALLOWED_CONTENT_TYPES = {"text/plain", "text/markdown", "application/pdf"}


def _verify_blob(meta: BlobStat, declared_mime: str, max_bytes: int) -> IngestErrorCode | None:
    """Check HEAD results against the declared upload. None means acceptable."""
    if meta.size > max_bytes:
        return "size_exceeded"
    if meta.size == 0:
        return "empty_content"
    if meta.content_type != declared_mime:
        return "type_mismatch"
    return None


def fail_document_or_404(
    doc_store: DocumentStorePort, document_id: str, error_code: IngestErrorCode
) -> DocumentStatusView:
    """Record a typed ingest failure, or 404 when the document is unknown."""
    failed = doc_store.fail(document_id, error_code)
    if failed is None:
        raise HTTPException(status_code=404, detail="unknown document")
    return failed


def initiate_upload(
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


def complete_upload(
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
        return fail_document_or_404(doc_store, document_id, "upload_incomplete")
    code = _verify_blob(meta, doc.mime, max_upload_bytes)
    if code is not None:
        return fail_document_or_404(doc_store, document_id, code)
    uploaded = doc_store.mark_uploaded(document_id)
    if uploaded is None:
        raise HTTPException(status_code=404, detail="unknown document")
    return uploaded
