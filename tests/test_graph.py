"""Graph seams: QA flow, ingest flow, and typed generation over fakes."""

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from rag_lab.adapters.documents import InMemoryDocumentStore
from rag_lab.adapters.embeddings import EmbeddingPort, StubEmbedding
from rag_lab.adapters.generation import complete, generate_answer, grounded_prompt
from rag_lab.adapters.llm_openai import LlmPort, StubLlm
from rag_lab.adapters.retrieval import InMemoryRetrieval, RetrievalPort
from rag_lab.adapters.storage import InMemoryStorage
from rag_lab.core.models import Chunk, Document
from rag_lab.graph.ingest import run_ingest_one
from rag_lab.graph.qa import QaGenerationError, QaOutcome, run_qa
from rag_lab.ingest.loaders import LoaderPort, PdfLoader, TextLoader

LOADERS: list[LoaderPort] = [TextLoader(), PdfLoader()]


def _seed() -> InMemoryRetrieval:
    return InMemoryRetrieval(
        [
            Chunk(
                id="chunk-cat",
                content="the cat sits on the mat",
                title="cat.txt",
                embedding=[1.0, 0.0, 0.0],
            ),
            Chunk(
                id="chunk-quantum",
                content="intro to quantum entanglement for beginners",
                title="quantum.txt",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )


def _run_qa(
    retriever: RetrievalPort | None = None,
    llm: LlmPort | None = None,
    embedder: EmbeddingPort | None = None,
) -> QaOutcome:
    return run_qa(
        embedder=embedder or StubEmbedding(dim=3),
        retriever=retriever if retriever is not None else _seed(),
        llm=llm or StubLlm("cited stub answer"),
        query="what is quantum entanglement?",
        retrieval_candidates=40,
        answer_context_chunks=5,
    )


def test_grounded_prompt_numbers_contexts() -> None:
    prompt = grounded_prompt("why?", ["first passage", "second passage"])
    assert "[1] first passage" in prompt
    assert "[2] second passage" in prompt
    assert "why?" in prompt


def test_grounded_prompt_empty_notes_no_passages() -> None:
    prompt = grounded_prompt("why?", [])
    assert "No knowledge-base passages" in prompt


def test_generate_answer_returns_typed_answer() -> None:
    answered = generate_answer(StubLlm("hi there"), "why?", ["some context"])
    assert answered.answer == "hi there"


def test_complete_empty_answer_fails_loudly() -> None:
    with pytest.raises(UnexpectedModelBehavior):
        complete(StubLlm(""), "prompt")


def test_run_qa_returns_trace_fields_and_answer() -> None:
    outcome = _run_qa()
    assert outcome["answer"] == "cited stub answer"
    assert len(outcome["candidates"]) == 2
    assert outcome["reranked"][0].chunk.id == "chunk-quantum"
    assert "quantum entanglement" in outcome["prompt"]


def test_run_qa_empty_kb_answers_without_context() -> None:
    outcome = _run_qa(retriever=InMemoryRetrieval())
    assert outcome["candidates"] == []
    assert outcome["reranked"] == []
    assert "No knowledge-base passages" in outcome["prompt"]


class _FailingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedder down")


class _FailingLlm:
    def chat(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("llm down")


def test_run_qa_embed_failure_propagates() -> None:
    with pytest.raises(RuntimeError, match="embedder down"):
        _run_qa(embedder=_FailingEmbedder())


def test_run_qa_llm_failure_becomes_generation_error() -> None:
    with pytest.raises(QaGenerationError):
        _run_qa(llm=_FailingLlm())


def test_run_ingest_one_idle_returns_none() -> None:
    assert (
        run_ingest_one(InMemoryDocumentStore(), InMemoryStorage(), StubEmbedding(), LOADERS) is None
    )


def test_run_ingest_one_missing_blob_marks_failed() -> None:
    docs = InMemoryDocumentStore()
    document_id = docs.initiate(
        Document(source="note.txt", title="note.txt", mime="text/plain", blob_url="memory://gone")
    )
    docs.mark_uploaded(document_id)
    assert run_ingest_one(docs, InMemoryStorage(), StubEmbedding(), LOADERS) == document_id
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "blob_missing")
