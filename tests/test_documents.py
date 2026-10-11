"""DocumentStore contract: fake behaves like Postgres over the ingest lifecycle."""

import os
import uuid

import pytest
from sqlmodel import Session, select

from rag_lab.adapters.db import get_engine
from rag_lab.adapters.documents import (
    DocumentStorePort,
    InMemoryDocumentStore,
    PostgresDocumentStore,
)
from rag_lab.adapters.tables import ChunkRow, DocumentRow
from rag_lab.core.config import Settings
from rag_lab.core.models import Chunk, Document

LIVE = os.environ.get("RAGLAB_LIVE_DB") == "1"
needs_live = pytest.mark.skipif(not LIVE, reason="needs live postgres (RAGLAB_LIVE_DB=1)")


def _document() -> Document:
    return Document(source="note.txt", title="note.txt", mime="text/plain", blob_url="memory://x")


def _chunks() -> list[Chunk]:
    return [Chunk(content="hello rag", title="note.txt", mime="text/plain")]


def lifecycle_roundtrip(store: DocumentStorePort) -> None:
    document_id = store.initiate(_document())
    view = store.get_status(document_id)
    assert view is not None
    assert view.status == "initiated"
    assert view.chunk_count == 0
    assert view.error_code is None
    assert store.get_document(document_id) is not None

    uploaded = store.mark_uploaded(document_id)
    assert uploaded is not None
    assert uploaded.status == "uploaded"

    claimed = store.claim_next()
    assert claimed is not None
    assert claimed.id == document_id
    assert store.claim_next() is None

    store.mark_embedding(document_id)
    status = store.get_status(document_id)
    assert status is not None
    assert status.status == "embedding"

    chunk = _chunks()[0].model_copy(update={"embedding": [0.25] * 768})
    store.finish(document_id, [chunk])
    done = store.get_status(document_id)
    assert done is not None
    assert done.status == "finished"
    assert done.chunk_count == 1


def test_in_memory_lifecycle_roundtrip() -> None:
    lifecycle_roundtrip(InMemoryDocumentStore())


def test_in_memory_complete_is_idempotent() -> None:
    store = InMemoryDocumentStore()
    document_id = store.initiate(_document())
    view = store.mark_uploaded(document_id)
    assert view is not None
    assert view.status == "uploaded"
    again = store.mark_uploaded(document_id)
    assert again is not None
    assert again.status == "uploaded"


def test_in_memory_fail_pairs_terminal_status_with_code() -> None:
    store = InMemoryDocumentStore()
    document_id = store.initiate(_document())
    view = store.fail(document_id, "blob_missing")
    assert view is not None
    assert view.status == "failed"
    assert view.error_code == "blob_missing"


def test_in_memory_unknown_ids_return_none() -> None:
    store = InMemoryDocumentStore()
    assert store.get_status(str(uuid.uuid4())) is None
    assert store.get_document(str(uuid.uuid4())) is None
    assert store.mark_uploaded(str(uuid.uuid4())) is None
    assert store.fail(str(uuid.uuid4()), "blob_missing") is None
    with pytest.raises(ValueError, match="unknown document"):
        store.finish(str(uuid.uuid4()), _chunks())


def test_in_memory_finish_rejects_empty_chunks() -> None:
    store = InMemoryDocumentStore()
    document_id = store.initiate(_document())
    with pytest.raises(ValueError, match="chunks must not be empty"):
        store.finish(document_id, [])


def test_postgres_finish_rejects_empty_chunks_without_touching_db() -> None:
    # The empty check precedes any IO, so no live database is needed.
    engine = get_engine("postgresql://rag:rag@localhost:5432/raglab")
    try:
        with pytest.raises(ValueError, match="chunks must not be empty"):
            PostgresDocumentStore(engine).finish(str(uuid.uuid4()), [])
    finally:
        engine.dispose()


@needs_live
def test_postgres_lifecycle_roundtrip() -> None:
    engine = get_engine(Settings().database_url)
    try:
        store = PostgresDocumentStore(engine)
        lifecycle_roundtrip(store)
        with Session(engine) as session:
            rows = list(session.exec(select(DocumentRow)).all())
            assert any(row.status == "finished" for row in rows)
            for row in rows:
                for chunk in session.exec(
                    select(ChunkRow).where(ChunkRow.document_id == row.id)
                ).all():
                    session.delete(chunk)
                session.delete(row)
            session.commit()
    finally:
        engine.dispose()


@needs_live
def test_postgres_finish_persists_768_dim_vectors() -> None:
    engine = get_engine(Settings().database_url)
    try:
        store = PostgresDocumentStore(engine)
        document_id = store.initiate(_document())
        store.mark_uploaded(document_id)
        assert store.claim_next() is not None
        chunk = _chunks()[0].model_copy(update={"embedding": [0.1] * 768})
        store.finish(document_id, [chunk])
        with Session(engine) as session:
            rows = list(
                session.exec(
                    select(ChunkRow).where(ChunkRow.document_id == uuid.UUID(document_id))
                ).all()
            )
            assert len(rows) == 1
            assert rows[0].embedding is not None
            assert len(rows[0].embedding or []) == 768
            for row in rows:
                session.delete(row)
            doc = session.get(DocumentRow, uuid.UUID(document_id))
            if doc is not None:
                session.delete(doc)
            session.commit()
    finally:
        engine.dispose()
