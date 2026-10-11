"""Contracts introduced by the review fix pass: no wedged docs, no orphan URLs."""

import pytest
from fastapi.testclient import TestClient

from rag_lab.adapters.documents import DocumentStorePort, InMemoryDocumentStore
from rag_lab.adapters.embeddings import EmbeddingPort, StubEmbedding
from rag_lab.adapters.llm_openai import LlmPort, StubLlm
from rag_lab.adapters.retrieval import InMemoryRetrieval, RetrievalPort
from rag_lab.adapters.storage import InMemoryStorage, StoragePort
from rag_lab.adapters.traces import InMemoryTraceStore, TraceStorePort
from rag_lab.api.main import create_app
from rag_lab.core.config import Settings
from rag_lab.core.models import Document, DocumentStatusView
from rag_lab.core.text import tokenize
from rag_lab.eval.run import SuiteSummary, evaluate_item
from rag_lab.graph.ingest import run_ingest_one
from rag_lab.ingest.loaders import LoaderPort, PdfLoader, TextLoader

LOADERS: list[LoaderPort] = [TextLoader(), PdfLoader()]


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
        max_upload_bytes=1024,
    )


def test_fakes_satisfy_ports() -> None:
    assert isinstance(InMemoryStorage(), StoragePort)
    assert isinstance(InMemoryDocumentStore(), DocumentStorePort)
    assert isinstance(InMemoryTraceStore(), TraceStorePort)
    assert isinstance(InMemoryRetrieval(), RetrievalPort)
    assert isinstance(StubEmbedding(), EmbeddingPort)
    assert isinstance(StubLlm(), LlmPort)
    assert isinstance(TextLoader(), LoaderPort)
    assert isinstance(PdfLoader(), LoaderPort)


def test_tokenize_shared_helper() -> None:
    assert tokenize("Hello, RAG-World 123!") == ["hello", "rag", "world", "123"]


def test_failed_status_requires_error_code() -> None:
    with pytest.raises(ValueError, match="requires an error_code"):
        DocumentStatusView(document_id="d", status="failed", chunk_count=0)


def test_non_failed_status_rejects_error_code() -> None:
    with pytest.raises(ValueError, match="only failed status"):
        DocumentStatusView(
            document_id="d", status="finished", chunk_count=1, error_code="embed_failed"
        )


def test_empty_suite_summary_is_zero() -> None:
    summary = SuiteSummary()
    assert summary.retrieval_hit_rate == 0.0
    assert summary.rerank_top1_rate == 0.0
    assert summary.mean_keyword_coverage == 0.0
    assert summary.mean_groundedness == 0.0
    assert not summary.passes()


def test_evaluate_item_malformed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="string id/query"):
        evaluate_item({"id": "x"})


def test_text_loader_handles_uppercase_extension() -> None:
    assert TextLoader().can_handle("NOTE.TXT", "application/octet-stream")
    assert TextLoader().can_handle("README.MD", "application/octet-stream")


class _RaisingDownloadStorage(InMemoryStorage):
    def download(self, name: str) -> bytes | None:
        raise RuntimeError("blob down")


class _FailingClaimStore(InMemoryDocumentStore):
    def claim_next(self) -> Document | None:
        raise RuntimeError("db down")


def _stage_uploaded(docs: InMemoryDocumentStore) -> str:
    document_id = docs.initiate(
        Document(source="note.txt", title="note.txt", mime="text/plain", blob_url="memory://gone")
    )
    docs.mark_uploaded(document_id)
    return document_id


def test_download_raise_maps_to_persist_failed() -> None:
    docs = InMemoryDocumentStore()
    document_id = _stage_uploaded(docs)
    assert run_ingest_one(docs, _RaisingDownloadStorage(), StubEmbedding(), LOADERS) == document_id
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "persist_failed")


def test_claim_failure_returns_none_without_raise() -> None:
    assert run_ingest_one(_FailingClaimStore(), InMemoryStorage(), StubEmbedding(), LOADERS) is None


class _FailingInitiateStore(InMemoryDocumentStore):
    def initiate(self, document: Document) -> str:
        raise RuntimeError("db down")


class _CountingStorage(InMemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.issued = 0

    def issue_upload_url(
        self, name: str, content_type: str, expires_seconds: int = 900
    ) -> tuple[str, int]:
        self.issued += 1
        return super().issue_upload_url(name, content_type, expires_seconds)


class _FailingIssueStorage(InMemoryStorage):
    def issue_upload_url(
        self, name: str, content_type: str, expires_seconds: int = 900
    ) -> tuple[str, int]:
        raise RuntimeError("sas down")


def test_initiate_db_failure_mints_no_url() -> None:
    storage = _CountingStorage()
    client = TestClient(
        create_app(settings=_settings(), storage=storage, doc_store=_FailingInitiateStore())
    )
    response = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": "note.txt", "content_type": "text/plain", "size_bytes": 9},
    )
    assert response.status_code == 500
    assert storage.issued == 0


def test_issue_url_failure_fails_row_instead_of_orphaning() -> None:
    docs = InMemoryDocumentStore()
    client = TestClient(
        create_app(settings=_settings(), storage=_FailingIssueStorage(), doc_store=docs)
    )
    response = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": "note.txt", "content_type": "text/plain", "size_bytes": 9},
    )
    assert response.status_code == 500
    assert len(docs.documents) == 1
    document_id = next(iter(docs.documents))
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "upload_incomplete")
