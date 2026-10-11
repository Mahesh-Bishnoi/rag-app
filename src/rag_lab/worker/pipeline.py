"""Async ingest pipeline. The API initiates/uploads; this module drains them.

Each claimed document advances uploaded → chunking → embedding → finished,
one stage commit at a time, so a crash retries idempotently (re-download,
re-derive). Failures flip to failed with a typed error_code; provider and
DB causes are logged server-side only, never stored in the code.
"""

import logging

from rag_lab.adapters.documents import DocumentStorePort
from rag_lab.adapters.embeddings import EmbeddingPort
from rag_lab.adapters.storage import StoragePort
from rag_lab.core.models import Chunk
from rag_lab.ingest.chunking import split_chunks
from rag_lab.ingest.loaders import LoaderPort

logger = logging.getLogger(__name__)


def blob_name(blob_url: str) -> str:
    """Object key from a stored canonical blob URL (never a SAS URL)."""
    return blob_url.rsplit("/", 1)[-1].split("?", 1)[0]


def _select_loader(
    loaders: list[LoaderPort], filename: str, content_type: str
) -> LoaderPort | None:
    return next(
        (candidate for candidate in loaders if candidate.can_handle(filename, content_type)),
        None,
    )


def _embed(chunks: list[Chunk], embedder: EmbeddingPort) -> list[Chunk] | None:
    """One vector per chunk. None means the provider failed or misaligned."""
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


def process_one(
    doc_store: DocumentStorePort,
    storage: StoragePort,
    embedder: EmbeddingPort,
    loaders: list[LoaderPort],
) -> str | None:
    """Claim and ingest one uploaded document. Returns its id, or None if idle."""
    claimed = doc_store.claim_next()
    if claimed is None or claimed.id is None:
        return None
    document_id = claimed.id
    data = storage.download(blob_name(claimed.blob_url))
    if data is None:
        doc_store.fail(document_id, "blob_missing")
        return document_id
    loader = _select_loader(loaders, claimed.source, claimed.mime)
    if loader is None:
        doc_store.fail(document_id, "type_mismatch")
        return document_id
    try:
        loaded = loader.load(data, source=claimed.source)
    except (UnicodeDecodeError, ValueError):
        doc_store.fail(document_id, "decode_failed")
        return document_id
    chunks = split_chunks(loaded)
    if not chunks:
        doc_store.fail(document_id, "empty_content")
        return document_id
    doc_store.mark_embedding(document_id)
    embedded = _embed(chunks, embedder)
    if embedded is None:
        logger.exception("chunk embedding failed for %s", document_id)
        doc_store.fail(document_id, "embed_failed")
        return document_id
    try:
        doc_store.finish(document_id, embedded)
    except Exception:
        logger.exception("document finish failed for %s", document_id)
        doc_store.fail(document_id, "persist_failed")
    return document_id


def drain(
    doc_store: DocumentStorePort,
    storage: StoragePort,
    embedder: EmbeddingPort,
    loaders: list[LoaderPort],
) -> int:
    """Process uploaded documents until the queue is empty. Returns the count."""
    processed = 0
    while process_one(doc_store, storage, embedder, loaders) is not None:
        processed += 1
    return processed
