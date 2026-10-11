"""QA trace persistence behind a port (ADR-0015).

The chat path saves fail-open: a working answer stays a 200 even when the
trace write fails. Table rows never leave: callers pass Pydantic DTOs in,
get DTOs back. Two adapters justify the seam: Postgres for prod, in-memory
fake for hermetic tests.
"""

import logging
import uuid
from typing import Protocol, runtime_checkable

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, select

from rag_lab.adapters.tables import ChunkRow, TraceRow
from rag_lab.core.models import Chunk, Citation, QaTrace, RetrievalResult, Trace

logger = logging.getLogger(__name__)


@runtime_checkable
class TraceStorePort(Protocol):
    def save(self, trace: Trace, answer: str, citations: list[Citation]) -> None: ...
    def get(self, trace_id: str) -> QaTrace | None: ...


def _retrieved_json(trace: Trace) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": result.chunk.id or "",
            "vector_score": result.vector_score,
            "bm25_score": result.bm25_score,
            "fused_rank": result.fused_rank,
        }
        for result in trace.retrieved
    ]


def _citations_json(citations: list[Citation]) -> list[dict[str, object]]:
    return [citation.model_dump() for citation in citations]


def to_record(trace: Trace, answer: str, citations: list[Citation]) -> QaTrace:
    """Combine the retrieval trace with its final answer. Pure."""
    return QaTrace(
        trace_id=trace.trace_id,
        query=trace.query,
        retrieved=trace.retrieved,
        rerank_order=trace.rerank_order,
        prompt=trace.prompt,
        answer=answer,
        citations=citations,
    )


class InMemoryTraceStore:
    """Test fake sharing the TraceStorePort interface."""

    def __init__(self) -> None:
        self._records: dict[str, QaTrace] = {}

    def save(self, trace: Trace, answer: str, citations: list[Citation]) -> None:
        self._records[trace.trace_id] = to_record(trace, answer, citations)

    def get(self, trace_id: str) -> QaTrace | None:
        return self._records.get(trace_id)


class PostgresTraceStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save(self, trace: Trace, answer: str, citations: list[Citation]) -> None:
        with Session(self._engine) as session:
            row = session.get(TraceRow, trace.trace_id)
            if row is None:
                row = TraceRow(trace_id=trace.trace_id, query=trace.query)
                session.add(row)
            row.query = trace.query
            row.retrieved = _retrieved_json(trace)
            row.rerank_order = list(trace.rerank_order)
            row.prompt = trace.prompt
            row.answer = answer
            row.citations = _citations_json(citations)
            session.add(row)
            session.commit()
            logger.debug("saved trace %s", trace.trace_id)

    def get(self, trace_id: str) -> QaTrace | None:
        with Session(self._engine) as session:
            row = session.get(TraceRow, trace_id)
            if row is None:
                return None
            return QaTrace(
                trace_id=row.trace_id,
                query=row.query,
                retrieved=self._hydrate(session, row.retrieved),
                rerank_order=list(row.rerank_order),
                prompt=row.prompt,
                answer=row.answer,
                citations=[Citation.model_validate(item) for item in row.citations],
            )

    @staticmethod
    def _hydrate(session: Session, stored: list[dict[str, object]]) -> list[RetrievalResult]:
        """Reattach chunk content for stored ids; missing rows keep id + scores."""
        ids = [str(item.get("chunk_id", "")) for item in stored]
        keys = []
        for chunk_id in ids:
            try:
                keys.append(uuid.UUID(chunk_id))
            except ValueError:
                continue
        by_id: dict[str, ChunkRow] = {}
        if keys:
            for found in session.exec(
                select(ChunkRow).where(SQLModel.metadata.tables["chunks"].c.id.in_(keys))
            ).all():
                if found.id is not None:
                    by_id[str(found.id)] = found
        results: list[RetrievalResult] = []
        for item in stored:
            chunk_id = str(item.get("chunk_id", ""))
            match = by_id.get(chunk_id)
            chunk = (
                Chunk(
                    id=chunk_id,
                    document_id=str(match.document_id),
                    content=match.content,
                    title=match.title,
                    heading_path=match.heading_path,
                    page=match.page,
                    mime=match.mime,
                )
                if match is not None
                else Chunk(id=chunk_id, content="(chunk content unavailable)")
            )
            vector = item.get("vector_score")
            bm25 = item.get("bm25_score")
            rank = item.get("fused_rank")
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    vector_score=float(vector) if isinstance(vector, (int, float)) else None,
                    bm25_score=float(bm25) if isinstance(bm25, (int, float)) else None,
                    fused_rank=int(rank) if isinstance(rank, int) else None,
                )
            )
        return results
