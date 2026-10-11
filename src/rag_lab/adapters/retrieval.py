"""Hybrid retrieval behind a port: vector cosine + BM25 keyword, fused with RRF.

Postgres is the prod adapter (pgvector `<=>` + `ts_rank`, HNSW index from
migration 0002); InMemoryRetrieval is the hermetic fake. Both return
RetrievalResults ordered best-first with 1-based fused_rank.
"""

import math
import uuid
from collections import Counter
from typing import Protocol, runtime_checkable

from sqlalchemy import Engine, text
from sqlmodel import Session, SQLModel, select

from rag_lab.adapters.tables import ChunkRow
from rag_lab.core.models import Chunk, RetrievalResult
from rag_lab.core.text import tokenize

RRF_K = 60


def validate_limit(limit: int) -> None:
    """Shared guard: search and rerank limits must be positive."""
    if limit <= 0:
        raise ValueError("limit must be positive")


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity in [0, 1] for non-negative embeddings. Loud on dim drift."""
    if len(left) != len(right):
        raise ValueError(f"vector dim mismatch {len(left)} != {len(right)}")
    if not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def bm25_scores(query: list[str], docs: list[list[str]]) -> list[float]:
    """Plain BM25 (k1=1.2, b=0.75) for the in-memory fake. Pure."""
    if not docs:
        return []
    doc_count = len(docs)
    doc_freq = Counter(term for doc in docs for term in set(doc))
    avg_len = sum(len(doc) for doc in docs) / doc_count
    scores: list[float] = []
    for doc in docs:
        term_freq = Counter(doc)
        doc_len = len(doc)
        total = 0.0
        for term in query:
            count = term_freq.get(term, 0)
            if count == 0:
                continue
            idf = math.log(1.0 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = 1.2 * (1.0 - 0.75 + 0.75 * (doc_len / avg_len if avg_len else 1.0))
            total += idf * (count * 2.2) / (count + norm)
        scores.append(total)
    return scores


def rrf_fuse(ranked: list[list[str]], limit: int) -> list[tuple[str, int]]:
    """Fuse ranked id lists with reciprocal-rank fusion.

    Returns (chunk_id, 1-based fused rank) best-first; ties break on id so
    fusion is deterministic.
    """
    validate_limit(limit)
    fused: dict[str, float] = {}
    for ids in ranked:
        for rank, chunk_id in enumerate(ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(chunk_id, rank) for rank, (chunk_id, _) in enumerate(ordered[:limit], start=1)]


@runtime_checkable
class RetrievalPort(Protocol):
    def search(
        self, query_embedding: list[float], query_text: str, limit: int
    ) -> list[RetrievalResult]: ...


class InMemoryRetrieval:
    """Test fake sharing the RetrievalPort interface. Seed via add()."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = list(chunks) if chunks else []

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def search(
        self, query_embedding: list[float], query_text: str, limit: int
    ) -> list[RetrievalResult]:
        validate_limit(limit)
        if not self._chunks:
            return []
        ids = [chunk.id or f"mem-{index}" for index, chunk in enumerate(self._chunks)]
        dim = len(query_embedding)
        vector_scores = [
            cosine(query_embedding, _zero_fill_missing_embedding(chunk.embedding, dim))
            for chunk in self._chunks
        ]
        keyword_scores = bm25_scores(
            tokenize(query_text), [tokenize(chunk.content) for chunk in self._chunks]
        )
        return _assemble(
            chunks=self._chunks,
            ids=ids,
            vector_ranked=_ranked_ids(ids, vector_scores),
            keyword_ranked=_ranked_ids(ids, keyword_scores),
            vector_scores=dict(zip(ids, vector_scores, strict=True)),
            keyword_scores=dict(zip(ids, keyword_scores, strict=True)),
            limit=limit,
        )


class PostgresRetrieval:
    """Prod adapter: pgvector cosine + ts_rank, fused in Python."""

    def __init__(self, engine: Engine, expected_dim: int = 768) -> None:
        self._engine = engine
        self._expected_dim = expected_dim

    def search(
        self, query_embedding: list[float], query_text: str, limit: int
    ) -> list[RetrievalResult]:
        validate_limit(limit)
        if len(query_embedding) != self._expected_dim:
            raise ValueError(f"query dim {len(query_embedding)} != expected {self._expected_dim}")
        literal = "[" + ",".join(repr(float(value)) for value in query_embedding) + "]"
        with Session(self._engine) as session:
            vector_rows = (
                session.execute(
                    text(
                        "SELECT id::text AS chunk_id, "
                        "1 - (embedding <=> CAST(:qv AS vector)) AS score "
                        "FROM chunks WHERE embedding IS NOT NULL "
                        "ORDER BY embedding <=> CAST(:qv AS vector) LIMIT :k"
                    ),
                    {"qv": literal, "k": limit},
                )
                .mappings()
                .all()
            )
            vector_ranked = [str(row["chunk_id"]) for row in vector_rows]
            vector_scores = {str(row["chunk_id"]): float(row["score"]) for row in vector_rows}
            keyword_rows = (
                session.execute(
                    text(
                        "SELECT id::text AS chunk_id, "
                        "ts_rank(text_search, plainto_tsquery('english', :qt)) AS score "
                        "FROM chunks "
                        "WHERE text_search @@ plainto_tsquery('english', :qt) "
                        "ORDER BY score DESC LIMIT :k"
                    ),
                    {"qt": query_text, "k": limit},
                )
                .mappings()
                .all()
            )
            keyword_ranked = [str(row["chunk_id"]) for row in keyword_rows]
            keyword_scores = {str(row["chunk_id"]): float(row["score"]) for row in keyword_rows}
            fused = rrf_fuse([vector_ranked, keyword_ranked], limit)
            return self._load(session, fused, vector_scores, keyword_scores)

    def _load(
        self,
        session: Session,
        fused: list[tuple[str, int]],
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
    ) -> list[RetrievalResult]:
        keys = [uuid.UUID(chunk_id) for chunk_id, _ in fused]
        if not keys:
            return []
        pairs = session.exec(
            select(ChunkRow).where(SQLModel.metadata.tables["chunks"].c.id.in_(keys))
        ).all()
        by_id: dict[str, ChunkRow] = {}
        for row in pairs:
            if row.id is not None:
                by_id[str(row.id)] = row
        results: list[RetrievalResult] = []
        for chunk_id, rank in fused:
            found = by_id.get(chunk_id)
            if found is None:
                continue
            results.append(
                RetrievalResult(
                    chunk=Chunk(
                        id=chunk_id,
                        document_id=str(found.document_id),
                        content=found.content,
                        title=found.title,
                        heading_path=found.heading_path,
                        page=found.page,
                        mime=found.mime,
                        byte_start=found.byte_start,
                        byte_end=found.byte_end,
                    ),
                    vector_score=vector_scores.get(chunk_id),
                    bm25_score=keyword_scores.get(chunk_id),
                    fused_rank=rank,
                )
            )
        return results


def _zero_fill_missing_embedding(embedding: list[float] | None, dim: int) -> list[float]:
    return embedding if embedding is not None else [0.0] * dim


def _ranked_ids(ids: list[str], scores: list[float]) -> list[str]:
    return [
        chunk_id
        for chunk_id, _ in sorted(
            zip(ids, scores, strict=True), key=lambda pair: (-pair[1], pair[0])
        )
    ]


def _assemble(
    *,
    chunks: list[Chunk],
    ids: list[str],
    vector_ranked: list[str],
    keyword_ranked: list[str],
    vector_scores: dict[str, float],
    keyword_scores: dict[str, float],
    limit: int,
) -> list[RetrievalResult]:
    fused = rrf_fuse([vector_ranked, keyword_ranked], limit)
    by_id = dict(zip(ids, chunks, strict=True))
    return [
        RetrievalResult(
            chunk=by_id[chunk_id].model_copy(update={"id": chunk_id}),
            vector_score=vector_scores.get(chunk_id),
            bm25_score=keyword_scores.get(chunk_id),
            fused_rank=rank,
        )
        for chunk_id, rank in fused
    ]
