"""Document persistence behind a port.

The upload endpoint orchestrates (validate → load → put blob → save);
this module owns the save step. Table rows never leave: callers pass
Pydantic DTOs in, get a document id back. Two adapters justify the seam:
Postgres for prod, in-memory fake for hermetic tests.
"""

import logging
import uuid
from typing import Protocol

from sqlalchemy import Engine
from sqlmodel import Session

from rag_lab.adapters.tables import ChunkRow, DocumentRow
from rag_lab.core.models import Chunk, Document

logger = logging.getLogger(__name__)


class DocumentStorePort(Protocol):
    def save(self, document: Document, chunks: list[Chunk]) -> str: ...


class InMemoryDocumentStore:
    """Test fake sharing the DocumentStorePort interface."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, list[Chunk]] = {}

    def save(self, document: Document, chunks: list[Chunk]) -> str:
        if not chunks:
            raise ValueError("chunks must not be empty")
        document_id = str(uuid.uuid4())
        self.documents[document_id] = document.model_copy(update={"id": document_id})
        self.chunks[document_id] = [
            chunk.model_copy(update={"id": str(uuid.uuid4()), "document_id": document_id})
            for chunk in chunks
        ]
        return document_id


class PostgresDocumentStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save(self, document: Document, chunks: list[Chunk]) -> str:
        if not chunks:
            raise ValueError("chunks must not be empty")
        with Session(self._engine) as session:
            doc_row = DocumentRow(
                source=document.source,
                title=document.title,
                mime=document.mime,
                blob_url=document.blob_url,
            )
            session.add(doc_row)
            session.flush()
            session.refresh(doc_row)
            doc_id = doc_row.id
            if doc_id is None:
                raise RuntimeError("document id missing after flush")
            for chunk in chunks:
                session.add(
                    ChunkRow(
                        document_id=doc_id,
                        content=chunk.content,
                        title=chunk.title,
                        heading_path=chunk.heading_path,
                        page=chunk.page,
                        mime=chunk.mime,
                        byte_start=chunk.byte_start,
                        byte_end=chunk.byte_end,
                        embedding=chunk.embedding,
                    )
                )
            session.commit()
            logger.debug("saved document %s with %d chunks", doc_id, len(chunks))
            return str(doc_id)
