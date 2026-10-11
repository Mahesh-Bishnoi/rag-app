"""Hybrid retrieval + heuristic rerank: fusion math, fake behavior, live SQL."""

import os
import uuid

import pytest
from sqlmodel import Session, select

from rag_lab.adapters.db import get_engine
from rag_lab.adapters.documents import PostgresDocumentStore
from rag_lab.adapters.rerank import rerank_candidates
from rag_lab.adapters.retrieval import (
    InMemoryRetrieval,
    PostgresRetrieval,
    bm25_scores,
    cosine,
    rrf_fuse,
    tokenize,
)
from rag_lab.adapters.tables import ChunkRow, DocumentRow
from rag_lab.core.config import Settings
from rag_lab.core.models import Chunk, Document, RetrievalResult

LIVE = os.environ.get("RAGLAB_LIVE_DB") == "1"
needs_live = pytest.mark.skipif(not LIVE, reason="needs live postgres (RAGLAB_LIVE_DB=1)")


def _chunk(content: str, embedding: list[float] | None = None, title: str = "note.txt") -> Chunk:
    return Chunk(id=str(uuid.uuid4()), content=content, title=title, embedding=embedding)


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Hello, RAG-World 123!") == ["hello", "rag", "world", "123"]


def test_cosine_identical_vectors_score_one() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dim mismatch"):
        cosine([1.0], [1.0, 0.0])


def test_bm25_ranks_exact_term_match_first() -> None:
    docs = [tokenize("quantum entanglement explained"), tokenize("weather report")]
    scores = bm25_scores(tokenize("quantum entanglement"), docs)
    assert scores[0] > scores[1]
    assert scores[1] == 0.0


def test_rrf_fuse_prefers_shared_hits_and_breaks_ties() -> None:
    fused = rrf_fuse([["a", "b"], ["b", "a"]], 2)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    assert [rank for _, rank in fused] == [1, 2]


def test_rrf_fuse_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        rrf_fuse([["a"]], 0)


def test_in_memory_search_fuses_vector_and_keyword() -> None:
    retriever = InMemoryRetrieval(
        [
            _chunk("unrelated words here", [1.0, 0.0], "vec.txt"),
            _chunk("query terms here query", [0.0, 1.0], "kw.txt"),
        ]
    )
    results = retriever.search([1.0, 0.0], "query terms", 2)
    assert [r.fused_rank for r in results] == [1, 2]
    assert all(r.vector_score is not None and r.bm25_score is not None for r in results)
    assert {r.chunk.title for r in results} == {"vec.txt", "kw.txt"}


def test_in_memory_search_empty_index_returns_empty() -> None:
    assert InMemoryRetrieval().search([1.0], "anything", 5) == []


def test_in_memory_search_rejects_bad_limit_and_dim_drift() -> None:
    retriever = InMemoryRetrieval([_chunk("hi", [1.0, 0.0])])
    with pytest.raises(ValueError, match="limit must be positive"):
        retriever.search([1.0, 0.0], "hi", 0)
    with pytest.raises(ValueError, match="dim mismatch"):
        retriever.search([1.0, 2.0, 3.0], "hi", 5)


def test_rerank_promotes_exact_phrase_match() -> None:
    query = "quantum entanglement"
    results = [
        RetrievalResult(chunk=_chunk("the cat sits", [1.0, 0.0]), fused_rank=1),
        RetrievalResult(chunk=_chunk("intro to quantum entanglement", [0.0, 1.0]), fused_rank=2),
    ]
    reranked = rerank_candidates(query, results, 2)
    assert reranked[0].chunk.content == "intro to quantum entanglement"
    assert reranked[0].fused_rank == 2


def test_rerank_truncates_and_rejects_bad_limit() -> None:
    results = [RetrievalResult(chunk=_chunk(f"doc {i}", [1.0])) for i in range(3)]
    assert len(rerank_candidates("doc", results, 2)) == 2
    with pytest.raises(ValueError, match="limit must be positive"):
        rerank_candidates("doc", results, 0)


def test_postgres_retrieval_rejects_dim_mismatch_without_io() -> None:
    engine = get_engine("postgresql://rag:rag@localhost:5432/raglab")
    try:
        retriever = PostgresRetrieval(engine, expected_dim=768)
        with pytest.raises(ValueError, match="query dim"):
            retriever.search([1.0, 0.0], "hi", 5)
    finally:
        engine.dispose()


@needs_live
def test_postgres_hybrid_search_finds_seeded_chunks() -> None:
    engine = get_engine(Settings().database_url)
    suffix = uuid.uuid4().hex[:8]
    alpha_title = f"alpha-{suffix}.txt"
    beta_title = f"beta-{suffix}.txt"
    try:
        store = PostgresDocumentStore(engine)
        alpha = Chunk(
            content="pgvector hybrid retrieval test alpha",
            title=alpha_title,
            embedding=[1.0] + [0.0] * 767,
        )
        beta = Chunk(
            content="unrelated weather report",
            title=beta_title,
            embedding=[0.0, 1.0] + [0.0] * 766,
        )
        document_id = store.initiate(
            Document(
                source="alpha.txt",
                title="alpha.txt",
                mime="text/plain",
                blob_url="memory://alpha.txt",
            )
        )
        store.mark_uploaded(document_id)
        assert store.claim_next() is not None
        store.finish(document_id, [alpha, beta])

        results = PostgresRetrieval(engine).search([1.0] + [0.0] * 767, "hybrid retrieval", 5)
        # The shared dev DB may hold rows from earlier runs; key on our titles.
        by_title = {r.chunk.title: r for r in results}
        assert by_title[alpha_title].fused_rank == 1
        assert by_title[alpha_title].vector_score is not None
        assert by_title[alpha_title].bm25_score is not None
        assert beta_title in by_title

        with Session(engine) as session:
            for row in session.exec(
                select(ChunkRow).where(ChunkRow.document_id == uuid.UUID(document_id))
            ).all():
                session.delete(row)
            doc = session.get(DocumentRow, uuid.UUID(document_id))
            if doc is not None:
                session.delete(doc)
            session.commit()
    finally:
        engine.dispose()
