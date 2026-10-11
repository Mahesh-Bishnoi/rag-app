from conftest import minimal_pdf
from fastapi.testclient import TestClient

from rag_lab.adapters.documents import InMemoryDocumentStore
from rag_lab.adapters.embeddings import StubEmbedding
from rag_lab.adapters.storage import InMemoryStorage
from rag_lab.api.main import create_app
from rag_lab.core.config import Settings
from rag_lab.ingest.loaders import PdfLoader, TextLoader
from rag_lab.worker.pipeline import drain


def make_client() -> tuple[TestClient, InMemoryDocumentStore, InMemoryStorage]:
    settings = Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
        max_upload_bytes=1024,
    )
    docs = InMemoryDocumentStore()
    storage = InMemoryStorage()
    client = TestClient(create_app(settings=settings, storage=storage, doc_store=docs))
    return client, docs, storage


def fake_put(storage: InMemoryStorage, upload_url: str, data: bytes, content_type: str) -> None:
    """Simulate the client's direct PUT to the signed URL (fake transport)."""
    name = upload_url.split("?", 1)[0].rsplit("/", 1)[-1]
    storage.put(name, data, content_type)


def initiate(
    client: TestClient, filename: str = "note.txt", mime: str = "text/plain", size: int = 9
) -> dict[str, object]:
    response = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": filename, "content_type": mime, "size_bytes": size},
    )
    assert response.status_code == 200
    body: dict[str, object] = response.json()
    assert body["document_id"]
    assert body["upload_url"]
    assert body["expires_in"] == 900
    return body


def test_health_shape() -> None:
    client, _, _ = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "db" in body
    assert "blob" in body


def test_models_lists_configured() -> None:
    client, _, _ = make_client()
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2


def test_chat_completions_honest_501() -> None:
    client, _, _ = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 501


def test_initiate_rejects_unsupported_type() -> None:
    client, _, _ = make_client()
    response = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": "evil.exe", "content_type": "application/x-msdownload", "size_bytes": 2},
    )
    assert response.status_code == 415


def test_initiate_rejects_oversize() -> None:
    client, _, _ = make_client()
    response = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": "big.txt", "content_type": "text/plain", "size_bytes": 1025},
    )
    assert response.status_code == 413


def test_initiate_rejects_empty_size() -> None:
    client, _, _ = make_client()
    response = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": "empty.txt", "content_type": "text/plain", "size_bytes": 0},
    )
    assert response.status_code == 422


def test_initiate_returns_signed_url_and_initiated_status() -> None:
    client, docs, _ = make_client()
    body = initiate(client)
    document_id = str(body["document_id"])
    assert str(body["upload_url"]).startswith("memory://")
    view = docs.get_status(document_id)
    assert view is not None
    assert view.status == "initiated"
    status = client.get(f"/v1/kb/documents/{document_id}").json()
    assert status["status"] == "initiated"
    assert status["chunk_count"] == 0
    assert status["error_code"] is None


def test_full_flow_text_put_complete_drain() -> None:
    client, docs, storage = make_client()
    body = initiate(client)
    document_id = str(body["document_id"])
    fake_put(storage, str(body["upload_url"]), b"hello rag", "text/plain")

    completed = client.post(f"/v1/kb/uploads/{document_id}/complete").json()
    assert completed["status"] == "uploaded"

    assert drain(docs, storage, StubEmbedding(), [TextLoader(), PdfLoader()]) == 1
    final = client.get(f"/v1/kb/documents/{document_id}").json()
    assert final["status"] == "finished"
    assert final["chunk_count"] == 1
    assert final["error_code"] is None
    assert docs.chunks[document_id][0].embedding == [0.0] * 768


def test_full_flow_pdf_put_complete_drain() -> None:
    client, docs, storage = make_client()
    body = initiate(client, filename="doc.pdf", mime="application/pdf", size=64)
    document_id = str(body["document_id"])
    fake_put(
        storage,
        str(body["upload_url"]),
        minimal_pdf(["hello pdf", "page two"]),
        "application/pdf",
    )
    completed = client.post(f"/v1/kb/uploads/{document_id}/complete").json()
    assert completed["status"] == "uploaded"
    assert drain(docs, storage, StubEmbedding(), [TextLoader(), PdfLoader()]) == 1
    final = client.get(f"/v1/kb/documents/{document_id}").json()
    assert final["status"] == "finished"
    assert final["chunk_count"] == 2


def test_complete_without_put_fails_with_upload_incomplete() -> None:
    client, _, _ = make_client()
    body = initiate(client)
    document_id = str(body["document_id"])
    completed = client.post(f"/v1/kb/uploads/{document_id}/complete").json()
    assert completed["status"] == "failed"
    assert completed["error_code"] == "upload_incomplete"


def test_complete_with_oversize_blob_fails_with_size_exceeded() -> None:
    client, _, storage = make_client()
    body = initiate(client)
    document_id = str(body["document_id"])
    fake_put(storage, str(body["upload_url"]), b"x" * 1025, "text/plain")
    completed = client.post(f"/v1/kb/uploads/{document_id}/complete").json()
    assert completed["status"] == "failed"
    assert completed["error_code"] == "size_exceeded"


def test_complete_with_wrong_content_type_fails_with_type_mismatch() -> None:
    client, _, storage = make_client()
    body = initiate(client)
    document_id = str(body["document_id"])
    fake_put(storage, str(body["upload_url"]), b"hello rag", "application/pdf")
    completed = client.post(f"/v1/kb/uploads/{document_id}/complete").json()
    assert completed["status"] == "failed"
    assert completed["error_code"] == "type_mismatch"


def test_complete_with_empty_blob_fails_with_empty_content() -> None:
    client, _, storage = make_client()
    body = initiate(client)
    document_id = str(body["document_id"])
    fake_put(storage, str(body["upload_url"]), b"", "text/plain")
    completed = client.post(f"/v1/kb/uploads/{document_id}/complete").json()
    assert completed["status"] == "failed"
    assert completed["error_code"] == "empty_content"


def test_complete_is_idempotent_after_uploaded() -> None:
    client, _, storage = make_client()
    body = initiate(client)
    document_id = str(body["document_id"])
    fake_put(storage, str(body["upload_url"]), b"hello rag", "text/plain")
    first = client.post(f"/v1/kb/uploads/{document_id}/complete").json()
    second = client.post(f"/v1/kb/uploads/{document_id}/complete").json()
    assert first["status"] == "uploaded"
    assert second["status"] == "uploaded"


def test_complete_unknown_document_is_404() -> None:
    client, _, _ = make_client()
    response = client.post("/v1/kb/uploads/00000000-0000-0000-0000-000000000000/complete")
    assert response.status_code == 404


def test_get_unknown_document_is_404() -> None:
    client, _, _ = make_client()
    response = client.get("/v1/kb/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


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
