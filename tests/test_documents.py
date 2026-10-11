"""DocumentStore contract: fake behaves like Postgres for the upload path."""

import os
import uuid

import pytest
from sqlmodel import Session, select

from rag_lab.adapters.db import get_engine
from rag_lab.adapters.documents import InMemoryDocumentStore, PostgresDocumentStore
from rag_lab.adapters.tables import ChunkRow, DocumentRow
from rag_lab.core.config import Settings
from rag_lab.core.models import Chunk, Document

LIVE = os.environ.get("RAGLAB_LIVE_DB") == "1"
needs_live = pytest.mark.skipif(not LIVE, reason="needs live postgres (RAGLAB_LIVE_DB=1)")


def _document() -> Document:
    return Document(source="note.txt", title="note.txt", mime="text/plain", blob_url="memory://x")


def _chunks() -> list[Chunk]:
    return [Chunk(content="hello rag", title="note.txt", mime="text/plain")]


def test_in_memory_save_assigns_ids() -> None:
    store = InMemoryDocumentStore()
    document_id = store.save(_document(), _chunks())
    assert store.documents[document_id].id == document_id
    assert store.documents[document_id].source == "note.txt"
    stored_chunks = store.chunks[document_id]
    assert len(stored_chunks) == 1
    assert stored_chunks[0].document_id == document_id
    assert stored_chunks[0].content == "hello rag"


def test_in_memory_save_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="chunks must not be empty"):
        InMemoryDocumentStore().save(_document(), [])


def test_in_memory_save_preserves_embeddings() -> None:
    store = InMemoryDocumentStore()
    chunk = _chunks()[0].model_copy(update={"embedding": [0.25] * 768})
    document_id = store.save(_document(), [chunk])
    assert store.chunks[document_id][0].embedding == [0.25] * 768


def test_postgres_save_rejects_empty_chunks() -> None:
    engine = get_engine("postgresql://rag:rag@localhost:5432/raglab")
    try:
        with pytest.raises(ValueError, match="chunks must not be empty"):
            PostgresDocumentStore(engine).save(_document(), [])
    finally:
        engine.dispose()


@needs_live
def test_postgres_save_roundtrip() -> None:
    engine = get_engine(Settings().database_url)
    try:
        document_id = PostgresDocumentStore(engine).save(_document(), _chunks())
        with Session(engine) as session:
            doc = session.get(DocumentRow, uuid.UUID(document_id))
            assert doc is not None
            assert doc.source == "note.txt"
            rows = list(session.exec(select(ChunkRow).where(ChunkRow.document_id == doc.id)).all())
            assert [row.content for row in rows] == ["hello rag"]
            for row in rows:
                session.delete(row)
            session.delete(doc)
            session.commit()
    finally:
        engine.dispose()


@needs_live
def test_postgres_save_roundtrip_with_embeddings() -> None:
    engine = get_engine(Settings().database_url)
    try:
        chunk = _chunks()[0].model_copy(update={"embedding": [0.1] * 768})
        document_id = PostgresDocumentStore(engine).save(_document(), [chunk])
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
