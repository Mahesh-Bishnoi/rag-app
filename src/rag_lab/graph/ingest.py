"""Ingest graph: LangGraph owns the download → load → split → embed → finish flow.

One linear chain of named nodes; a typed _IngestFailure aborts the run and
the runner maps it to doc_store.fail with the same error codes the pipeline
used. Stage commits keep their order (mark_embedding before embed, finish
last) so a crash still retries idempotently.
"""

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_lab.adapters.documents import DocumentStorePort
from rag_lab.adapters.embeddings import EmbeddingPort
from rag_lab.adapters.storage import StoragePort, blob_name
from rag_lab.core.models import Chunk, IngestErrorCode
from rag_lab.ingest.chunking import split_chunks
from rag_lab.ingest.loaders import LoaderPort

logger = logging.getLogger(__name__)


class IngestState(TypedDict, total=False):
    document_id: str
    blob_url: str
    source: str
    mime: str
    data: bytes
    loaded: list[Chunk]
    chunks: list[Chunk]


class _IngestFailure(Exception):
    def __init__(self, code: IngestErrorCode) -> None:
        super().__init__(code)
        self.code = code


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


Graph = StateGraph[IngestState, None, IngestState, IngestState]


def _add_prepare_nodes(graph: Graph, storage: StoragePort, loaders: list[LoaderPort]) -> None:
    """Download → load → split nodes: bytes in, unembedded chunks out."""

    def _download(state: IngestState) -> dict[str, object]:
        data = storage.download(blob_name(state["blob_url"]))
        if data is None:
            raise _IngestFailure("blob_missing")
        return {"data": data}

    def _load(state: IngestState) -> dict[str, object]:
        loader = _select_loader(loaders, state["source"], state["mime"])
        if loader is None:
            raise _IngestFailure("type_mismatch")
        try:
            loaded = loader.load(state["data"], source=state["source"])
        except (UnicodeDecodeError, ValueError):
            raise _IngestFailure("decode_failed") from None
        return {"loaded": loaded}

    def _split(state: IngestState) -> dict[str, object]:
        chunks = split_chunks(state["loaded"])
        if not chunks:
            raise _IngestFailure("empty_content")
        return {"chunks": chunks}

    graph.add_node("download", _download)
    graph.add_node("load", _load)
    graph.add_node("split", _split)


def _add_persist_nodes(graph: Graph, doc_store: DocumentStorePort, embedder: EmbeddingPort) -> None:
    """Embed → finish nodes: chunks in, finished document out."""

    def _embed_chunks(state: IngestState) -> dict[str, object]:
        doc_store.mark_embedding(state["document_id"])
        embedded = _embed(state["chunks"], embedder)
        if embedded is None:
            logger.exception("chunk embedding failed for %s", state["document_id"])
            raise _IngestFailure("embed_failed")
        return {"chunks": embedded}

    def _finish(state: IngestState) -> dict[str, object]:
        try:
            doc_store.finish(state["document_id"], state["chunks"])
        except Exception:
            logger.exception("document finish failed for %s", state["document_id"])
            raise _IngestFailure("persist_failed") from None
        return {}

    graph.add_node("embed", _embed_chunks)
    graph.add_node("finish", _finish)


def build_ingest_graph(
    doc_store: DocumentStorePort,
    storage: StoragePort,
    embedder: EmbeddingPort,
    loaders: list[LoaderPort],
) -> CompiledStateGraph[IngestState, None, IngestState, IngestState]:
    """Wire the five ingest stages in order. Ports are injected, never constructed."""
    graph: Graph = StateGraph(IngestState)
    _add_prepare_nodes(graph, storage, loaders)
    _add_persist_nodes(graph, doc_store, embedder)
    graph.set_entry_point("download")
    graph.add_edge("download", "load")
    graph.add_edge("load", "split")
    graph.add_edge("split", "embed")
    graph.add_edge("embed", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def run_ingest_one(
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
    try:
        build_ingest_graph(doc_store, storage, embedder, loaders).invoke(
            {
                "document_id": document_id,
                "blob_url": claimed.blob_url,
                "source": claimed.source,
                "mime": claimed.mime,
            }
        )
    except _IngestFailure as failure:
        doc_store.fail(document_id, failure.code)
    return document_id
