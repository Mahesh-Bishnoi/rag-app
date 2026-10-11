"""Shared Pydantic v2 schemas. Stage boundaries log these shapes."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str | None = None
    source: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=512)
    mime: str = Field(min_length=1, max_length=128)
    blob_url: str = Field(min_length=1, max_length=2048)


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
