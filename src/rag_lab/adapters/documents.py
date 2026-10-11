"""Document persistence behind a port.

Async ingest (ADR-0013): the API initiates documents and flips them to
uploaded on complete; the worker claims uploaded rows and advances
chunking → embedding → finished/failed, writing chunks in the same
transaction as the finished flip. Table rows never leave: callers pass
Pydantic DTOs in, get ids and status views back. Two adapters justify
the seam: Postgres for prod, in-memory fake for hermetic tests.
"""

import logging
import uuid
from typing import Protocol, cast

from sqlalchemy import Engine, func
from sqlmodel import Session, SQLModel, select

from rag_lab.adapters.tables import ChunkRow, DocumentRow
from rag_lab.core.models import (
    Chunk,
    Document,
    DocumentStatus,
    DocumentStatusView,
    IngestErrorCode,
)

logger = logging.getLogger(__name__)


class DocumentStorePort(Protocol):
    def initiate(self, document: Document) -> str: ...
    def get_document(self, document_id: str) -> Document | None: ...
    def mark_uploaded(self, document_id: str) -> DocumentStatusView | None: ...
    def fail(self, document_id: str, error_code: IngestErrorCode) -> DocumentStatusView | None: ...
    def claim_next(self) -> Document | None: ...
    def mark_embedding(self, document_id: str) -> None: ...
    def finish(self, document_id: str, chunks: list[Chunk]) -> None: ...
    def get_status(self, document_id: str) -> DocumentStatusView | None: ...


class InMemoryDocumentStore:
    """Test fake sharing the DocumentStorePort interface."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self.statuses: dict[str, DocumentStatus] = {}
        self.errors: dict[str, IngestErrorCode | None] = {}

    def initiate(self, document: Document) -> str:
        document_id = str(uuid.uuid4())
        self.documents[document_id] = document.model_copy(update={"id": document_id})
        self.chunks[document_id] = []
        self.statuses[document_id] = "initiated"
        self.errors[document_id] = None
        return document_id

    def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def mark_uploaded(self, document_id: str) -> DocumentStatusView | None:
        if document_id not in self.documents:
            return None
        if self.statuses[document_id] == "initiated":
            self.statuses[document_id] = "uploaded"
        return self.get_status(document_id)

    def fail(self, document_id: str, error_code: IngestErrorCode) -> DocumentStatusView | None:
        if document_id not in self.documents:
            return None
        self.statuses[document_id] = "failed"
        self.errors[document_id] = error_code
        return self.get_status(document_id)

    def claim_next(self) -> Document | None:
        for document_id, status in self.statuses.items():
            if status == "uploaded":
                self.statuses[document_id] = "chunking"
                return self.documents[document_id]
        return None

    def mark_embedding(self, document_id: str) -> None:
        if document_id in self.statuses:
            self.statuses[document_id] = "embedding"

    def finish(self, document_id: str, chunks: list[Chunk]) -> None:
        if document_id not in self.documents:
            raise ValueError(f"unknown document {document_id}")
        if not chunks:
            raise ValueError("chunks must not be empty")
        self.chunks[document_id] = [
            chunk.model_copy(update={"id": str(uuid.uuid4()), "document_id": document_id})
            for chunk in chunks
        ]
        self.statuses[document_id] = "finished"

    def get_status(self, document_id: str) -> DocumentStatusView | None:
        if document_id not in self.documents:
            return None
        return DocumentStatusView(
            document_id=document_id,
            status=self.statuses[document_id],
            chunk_count=len(self.chunks[document_id]),
            error_code=self.errors[document_id],
        )


def _view(
    doc_id: uuid.UUID, status: str, error_code: str | None, chunk_count: int
) -> DocumentStatusView:
    # Narrow at the persistence boundary: the CHECK constraint limits these
    # columns to the Literal members, but SQLModel types them as plain str.
    return DocumentStatusView(
        document_id=str(doc_id),
        status=cast("DocumentStatus", status),
        chunk_count=chunk_count,
        error_code=cast("IngestErrorCode | None", error_code),
    )


def _as_uuid(document_id: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(document_id)
    except ValueError:
        return None


class PostgresDocumentStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def initiate(self, document: Document) -> str:
        with Session(self._engine) as session:
            row = DocumentRow(
                source=document.source,
                title=document.title,
                mime=document.mime,
                blob_url=document.blob_url,
                status="initiated",
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("document id missing after commit")
            logger.debug("initiated document %s", row.id)
            return str(row.id)

    def get_document(self, document_id: str) -> Document | None:
        key = _as_uuid(document_id)
        if key is None:
            return None
        with Session(self._engine) as session:
            row = session.get(DocumentRow, key)
            if row is None or row.id is None:
                return None
            return Document(
                id=str(row.id),
                source=row.source,
                title=row.title,
                mime=row.mime,
                blob_url=row.blob_url,
            )

    def mark_uploaded(self, document_id: str) -> DocumentStatusView | None:
        key = _as_uuid(document_id)
        if key is None:
            return None
        with Session(self._engine) as session:
            row = session.get(DocumentRow, key)
            if row is None or row.id is None:
                return None
            if row.status == "initiated":
                row.status = "uploaded"
                session.add(row)
                session.commit()
                session.refresh(row)
            return _view(row.id, row.status, row.error_code, self._chunk_count(session, key))

    def fail(self, document_id: str, error_code: IngestErrorCode) -> DocumentStatusView | None:
        key = _as_uuid(document_id)
        if key is None:
            return None
        with Session(self._engine) as session:
            row = session.get(DocumentRow, key)
            if row is None or row.id is None:
                return None
            row.status = "failed"
            row.error_code = error_code
            session.add(row)
            session.commit()
            session.refresh(row)
            return _view(row.id, row.status, row.error_code, self._chunk_count(session, key))

    def claim_next(self) -> Document | None:
        with Session(self._engine) as session:
            row = session.exec(
                select(DocumentRow)
                .where(DocumentRow.status == "uploaded")
                .order_by(SQLModel.metadata.tables["documents"].c.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).first()
            if row is None or row.id is None:
                return None
            row.status = "chunking"
            session.add(row)
            session.commit()
            return Document(
                id=str(row.id),
                source=row.source,
                title=row.title,
                mime=row.mime,
                blob_url=row.blob_url,
            )

    def mark_embedding(self, document_id: str) -> None:
        key = _as_uuid(document_id)
        if key is None:
            return
        with Session(self._engine) as session:
            row = session.get(DocumentRow, key)
            if row is None:
                return
            row.status = "embedding"
            session.add(row)
            session.commit()

    def finish(self, document_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        key = _as_uuid(document_id)
        if key is None:
            raise ValueError(f"unknown document {document_id}")
        with Session(self._engine) as session:
            row = session.get(DocumentRow, key)
            if row is None:
                raise ValueError(f"unknown document {document_id}")
            for chunk in chunks:
                session.add(
                    ChunkRow(
                        document_id=key,
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
            row.status = "finished"
            session.add(row)
            session.commit()
            logger.debug("finished document %s with %d chunks", key, len(chunks))

    def get_status(self, document_id: str) -> DocumentStatusView | None:
        key = _as_uuid(document_id)
        if key is None:
            return None
        with Session(self._engine) as session:
            row = session.get(DocumentRow, key)
            if row is None or row.id is None:
                return None
            return _view(row.id, row.status, row.error_code, self._chunk_count(session, key))

    @staticmethod
    def _chunk_count(session: Session, key: uuid.UUID) -> int:
        result = session.exec(
            select(func.count()).select_from(ChunkRow).where(ChunkRow.document_id == key)
        ).first()
        return int(result or 0)
