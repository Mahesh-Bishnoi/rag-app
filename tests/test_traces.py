"""QA trace persistence: fake contract, API fail-open save, GET endpoint, live SQL."""

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from rag_lab.adapters.db import get_engine
from rag_lab.adapters.documents import InMemoryDocumentStore, PostgresDocumentStore
from rag_lab.adapters.embeddings import StubEmbedding
from rag_lab.adapters.llm_openai import StubLlm
from rag_lab.adapters.retrieval import InMemoryRetrieval
from rag_lab.adapters.storage import InMemoryStorage
from rag_lab.adapters.tables import ChunkRow, DocumentRow, TraceRow
from rag_lab.adapters.traces import InMemoryTraceStore, PostgresTraceStore, to_record
from rag_lab.api.main import create_app
from rag_lab.core.config import Settings
from rag_lab.core.models import Chunk, Citation, Document, RetrievalResult, Trace

LIVE = os.environ.get("RAGLAB_LIVE_DB") == "1"
needs_live = pytest.mark.skipif(not LIVE, reason="needs live postgres (RAGLAB_LIVE_DB=1)")


def _trace(trace_id: str = "trace-1") -> Trace:
    return Trace(
        trace_id=trace_id,
        query="what index?",
        retrieved=[
            RetrievalResult(
                chunk=Chunk(id="chunk-1", content="hnsw cosine index", title="note.txt"),
                vector_score=0.9,
                bm25_score=0.1,
                fused_rank=1,
            )
        ],
        rerank_order=["chunk-1"],
        prompt="Answer: what index?",
    )


def _citations() -> list[Citation]:
    return [Citation(chunk_id="chunk-1", source="note.txt", title="note.txt")]


def test_to_record_combines_trace_with_answer() -> None:
    record = to_record(_trace(), "hnsw", _citations())
    assert record.trace_id == "trace-1"
    assert record.query == "what index?"
    assert record.answer == "hnsw"
    assert [c.chunk_id for c in record.citations] == ["chunk-1"]
    assert [r.chunk.id for r in record.retrieved] == ["chunk-1"]


def test_in_memory_save_get_round_trip() -> None:
    store = InMemoryTraceStore()
    assert store.get("missing") is None
    store.save(_trace(), "hnsw", _citations())
    record = store.get("trace-1")
    assert record is not None
    assert (record.answer, record.prompt) == ("hnsw", "Answer: what index?")
    assert record.rerank_order == ["chunk-1"]
    store.save(_trace(), "updated", _citations())
    updated = store.get("trace-1")
    assert updated is not None
    assert updated.answer == "updated"


def _client(traces: object) -> TestClient:
    return TestClient(
        create_app(
            settings=Settings(
                database_url="postgresql://rag:rag@localhost:5432/raglab",
                azurite_connection_string="",
            ),
            storage=InMemoryStorage(),
            doc_store=InMemoryDocumentStore(),
            embedder=StubEmbedding(dim=3),
            retriever=InMemoryRetrieval(
                [
                    Chunk(
                        id="chunk-1",
                        content="hnsw cosine index",
                        title="note.txt",
                        embedding=[1.0, 0.0, 0.0],
                    )
                ]
            ),
            llm=StubLlm("traceable answer"),
            trace_store=traces,  # type: ignore[arg-type]
        )
    )


def test_chat_then_get_trace_returns_full_record() -> None:
    client = _client(InMemoryTraceStore())
    body = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "what index?"}]},
    ).json()
    assert str(body["id"]).startswith("chatcmpl-")
    trace_id = str(body["id"]).removeprefix("chatcmpl-")
    record = client.get(f"/v1/traces/{trace_id}").json()
    assert record["trace_id"] == trace_id
    assert record["query"] == "what index?"
    assert record["answer"] == "traceable answer"
    assert record["prompt"]
    assert [c["chunk_id"] for c in record["citations"]] == ["chunk-1"]
    assert record["retrieved"][0]["chunk"]["id"] == "chunk-1"
    assert record["retrieved"][0]["fused_rank"] == 1
    assert record["rerank_order"] == ["chunk-1"]


def test_get_trace_unknown_is_404() -> None:
    client = _client(InMemoryTraceStore())
    assert client.get("/v1/traces/nope").status_code == 404


class _FailingTraceStore:
    def save(self, trace: Trace, answer: str, citations: list[Citation]) -> None:
        raise RuntimeError("trace db down")

    def get(self, trace_id: str) -> None:
        return None


def test_chat_stays_200_when_trace_save_fails() -> None:
    client = _client(_FailingTraceStore())
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "traceable answer"


def test_worker_run_module_imports() -> None:
    import rag_lab.worker.run as run_module

    assert callable(run_module.main)


@needs_live
def test_postgres_trace_save_get_round_trip_with_hydration() -> None:
    engine = get_engine(Settings().database_url)
    suffix = uuid.uuid4().hex[:8]
    trace_id = f"live-trace-{suffix}"
    try:
        docs = PostgresDocumentStore(engine)
        document_id = docs.initiate(
            Document(
                source="note.txt",
                title="note.txt",
                mime="text/plain",
                blob_url="memory://note.txt",
            )
        )
        docs.mark_uploaded(document_id)
        assert docs.claim_next() is not None
        docs.finish(
            document_id,
            [Chunk(content="hnsw cosine index", title="note.txt", embedding=[1.0] + [0.0] * 767)],
        )
        with Session(engine) as session:
            row = session.exec(
                select(ChunkRow).where(ChunkRow.document_id == uuid.UUID(document_id))
            ).first()
            assert row is not None
            assert row.id is not None
            chunk_id = str(row.id)
        traces = PostgresTraceStore(engine)
        assert traces.get(trace_id) is None
        traces.save(
            Trace(
                trace_id=trace_id,
                query="what index?",
                retrieved=[
                    RetrievalResult(
                        chunk=Chunk(id=chunk_id, content="hnsw cosine index", title="note.txt"),
                        vector_score=0.9,
                        bm25_score=0.1,
                        fused_rank=1,
                    )
                ],
                rerank_order=[chunk_id],
                prompt="Answer: what index?",
            ),
            "hnsw",
            [Citation(chunk_id=chunk_id, source="note.txt", title="note.txt")],
        )
        record = traces.get(trace_id)
        assert record is not None
        assert (record.query, record.answer) == ("what index?", "hnsw")
        assert record.citations[0].chunk_id == chunk_id
        assert record.retrieved[0].chunk.content == "hnsw cosine index"
        assert record.retrieved[0].vector_score == pytest.approx(0.9)
        traces.save(_trace(trace_id), "updated", _citations())
        assert traces.get(trace_id) is not None
        assert (traces.get(trace_id) or record).answer == "updated"

        with Session(engine) as session:
            trace_row = session.get(TraceRow, trace_id)
            if trace_row is not None:
                session.delete(trace_row)
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
