"""Shared Pydantic v2 schemas. Stage boundaries log these shapes."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DocumentStatus = Literal["initiated", "uploaded", "chunking", "embedding", "finished", "failed"]

IngestErrorCode = Literal[
    "upload_incomplete",
    "blob_missing",
    "size_exceeded",
    "type_mismatch",
    "decode_failed",
    "empty_content",
    "embed_failed",
    "persist_failed",
]


class Document(BaseModel):
    id: str | None = None
    source: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=512)
    mime: str = Field(min_length=1, max_length=128)
    blob_url: str = Field(min_length=1, max_length=2048)


class DocumentStatusView(BaseModel):
    document_id: str
    status: DocumentStatus
    chunk_count: int = Field(ge=0)
    error_code: IngestErrorCode | None = None

    @model_validator(mode="after")
    def _check_status_error_pair(self) -> "DocumentStatusView":
        if self.status == "failed" and self.error_code is None:
            raise ValueError("failed status requires an error_code")
        if self.status != "failed" and self.error_code is not None:
            raise ValueError("only failed status may carry an error_code")
        return self


class Chunk(BaseModel):
    id: str | None = None
    document_id: str | None = None
    content: str = Field(min_length=1)
    title: str = ""
    heading_path: str = ""
    page: int | None = None
    mime: str = "text/plain"
    byte_start: int | None = None
    byte_end: int | None = None
    embedding: list[float] | None = None


class RetrievalResult(BaseModel):
    chunk: Chunk
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_rank: int | None = None


class Citation(BaseModel):
    chunk_id: str
    source: str
    title: str
    heading_path: str = ""
    page: int | None = None


class AnswerWithCitations(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str


class Trace(BaseModel):
    trace_id: str
    query: str
    retrieved: list[RetrievalResult] = Field(default_factory=list)
    rerank_order: list[str] = Field(default_factory=list)
    prompt: str = ""


class QaTrace(Trace):
    """Persisted per-question record: retrieval trace + final answer (ADR-0015)."""

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
