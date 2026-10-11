from fastapi.testclient import TestClient

from rag_lab.adapters.documents import InMemoryDocumentStore
from rag_lab.adapters.storage import InMemoryStorage
from rag_lab.api.main import create_app
from rag_lab.core.config import Settings
from rag_lab.core.models import Chunk, Document


def make_client() -> tuple[TestClient, InMemoryDocumentStore]:
    settings = Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
        max_upload_bytes=1024,
    )
    docs = InMemoryDocumentStore()
    client = TestClient(create_app(settings=settings, storage=InMemoryStorage(), doc_store=docs))
    return client, docs


def test_health_shape() -> None:
    client, _ = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "db" in body
    assert "blob" in body


def test_models_lists_configured() -> None:
    client, _ = make_client()
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2


def test_chat_completions_honest_501() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 501


def test_upload_rejects_unsupported_type() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/kb/upload",
        files={"file": ("evil.exe", b"MZ", "application/x-msdownload")},
    )
    assert response.status_code == 415


def test_upload_rejects_oversize() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/kb/upload",
        files={"file": ("big.txt", b"x" * 1025, "text/plain")},
    )
    assert response.status_code == 413


def test_upload_rejects_empty() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/kb/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422


def test_upload_roundtrip_text() -> None:
    client, docs = make_client()
    response = client.post(
        "/v1/kb/upload",
        files={"file": ("note.txt", b"hello rag", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["bytes"] == 9
    assert body["blob_url"].startswith("memory://")
    assert body["chunk_count"] == 1
    document_id = body["document_id"]
    assert document_id
    stored = docs.documents[document_id]
    assert stored.source == "note.txt"
    assert stored.title == "note.txt"
    assert stored.mime == "text/plain"
    assert stored.blob_url == body["blob_url"]
    assert [chunk.content for chunk in docs.chunks[document_id]] == ["hello rag"]


def test_upload_pdf_without_loader_is_415() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/kb/upload",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 415


def test_upload_rejects_non_utf8_text() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/kb/upload",
        files={"file": ("note.txt", b"\xff\xfe\x00bad", "text/plain")},
    )
    assert response.status_code == 422


def test_upload_returns_500_when_persist_fails() -> None:
    settings = Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
        max_upload_bytes=1024,
    )
    client = TestClient(
        create_app(settings=settings, storage=InMemoryStorage(), doc_store=_FailingStore())
    )
    response = client.post(
        "/v1/kb/upload",
        files={"file": ("note.txt", b"hello rag", "text/plain")},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "ingest failed"


class _FailingStore:
    def save(self, document: Document, chunks: list[Chunk]) -> str:
        raise RuntimeError("db down")


def test_health_falls_back_to_memory_without_connection_string() -> None:
    settings = Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
    )
    body = TestClient(create_app(settings=settings, storage=None)).get("/health").json()
    assert body["status"] == "ok"
    assert body["blob"] is True


def test_health_reports_blob_false_when_azurite_down() -> None:
    settings = Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        # Fake key: the probe hits a closed port, auth never happens.
        azurite_connection_string=(
            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
            "AccountKey=aGk=;"
            "BlobEndpoint=http://127.0.0.1:19999/devstoreaccount1;"
        ),
    )
    client = TestClient(create_app(settings=settings, storage=None))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["blob"] is False
