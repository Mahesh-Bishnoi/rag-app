"""Async ingest pipeline. The API initiates/uploads; this module drains them.

Stage flow and failure codes live in the LangGraph ingest graph
(graph/ingest.py), which is canonical: process_one is an explicit alias for
run_ingest_one kept for the worker-layer name, drain repeats until idle.
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
    """Alias for run_ingest_one: claim and ingest one document, None if idle."""
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
