"""SQLModel table rows. Storage schema only.

API and stage-boundary DTOs stay Pydantic in core/models.py and api/main.py:
table rows are never returned from endpoints, and DTOs are never persisted.
The two shapes evolve for different reasons (migration safety vs API
stability), so they are not shared, subclassed, or converted implicitly.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Column, Computed, DateTime, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Field, SQLModel


def _uuid_pk() -> Column[uuid.UUID]:
    return Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))


class DocumentRow(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('initiated', 'uploaded', 'chunking', 'embedding', 'finished', 'failed')",
            name="documents_status_check",
        ),
        Index("documents_claim_idx", "created_at", postgresql_where=text("status = 'uploaded'")),
    )

    id: uuid.UUID | None = Field(default=None, sa_column=_uuid_pk())
    source: str
    title: str
    mime: str
    blob_url: str
    status: str = "initiated"
    error_code: str | None = None
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("now()")),
    )


class ChunkRow(SQLModel, table=True):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("chunks_document_id_idx", "document_id"),
        Index("chunks_text_search_idx", "text_search", postgresql_using="gin"),
        # Mirrors migration 0002 (ADR-0014): cosine HNSW for the vector side
        # of hybrid retrieval. A model with dim != 768 needs a migration
        # changing vector(n) plus a Settings.embedding_dimensions bump.
        Index(
            "chunks_embedding_hnsw_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: uuid.UUID | None = Field(default=None, sa_column=_uuid_pk())
    document_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    content: str
    title: str = ""
    heading_path: str = ""
    page: int | None = None
    mime: str = "text/plain"
    byte_start: int | None = None
    byte_end: int | None = None
    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector(768)))
    text_search: str | None = Field(
        default=None,
        sa_column=Column(TSVECTOR(), Computed("to_tsvector('english', content)", persisted=True)),
    )


class TraceRow(SQLModel, table=True):
    """Persisted QA trace (ADR-0015). Mirrors migration 0003 exactly."""

    __tablename__ = "qa_traces"

    trace_id: str = Field(primary_key=True)
    query: str
    retrieved: list[dict[str, object]] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, server_default=text("'[]'"))
    )
    rerank_order: list[str] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, server_default=text("'[]'"))
    )
    prompt: str = Field(default="")
    answer: str = Field(default="")
    citations: list[dict[str, object]] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, server_default=text("'[]'"))
    )
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("now()")),
    )
