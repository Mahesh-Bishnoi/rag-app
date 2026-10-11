"""Async ingest pipeline. The API initiates/uploads; this module drains them.

Stage flow lives in the LangGraph ingest graph (graph/ingest.py):
process_one claims one uploaded document and runs it through the graph,
drain repeats until the queue is idle. Failure codes are unchanged
(blob_missing, type_mismatch, decode_failed, empty_content, embed_failed,
persist_failed).
"""

from rag_lab.adapters.documents import DocumentStorePort
from rag_lab.adapters.embeddings import EmbeddingPort
from rag_lab.adapters.storage import StoragePort
from rag_lab.graph.ingest import run_ingest_one
from rag_lab.ingest.loaders import LoaderPort


def process_one(
    doc_store: DocumentStorePort,
    storage: StoragePort,
    embedder: EmbeddingPort,
    loaders: list[LoaderPort],
) -> str | None:
    """Claim and ingest one uploaded document. Returns its id, or None if idle."""
    return run_ingest_one(doc_store, storage, embedder, loaders)


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
