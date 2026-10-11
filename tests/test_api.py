from fastapi.testclient import TestClient

from rag_lab.adapters.storage import InMemoryStorage
from rag_lab.api.main import create_app
from rag_lab.core.config import Settings


def make_client() -> TestClient:
    settings = Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
        max_upload_bytes=1024,
    )
    return TestClient(create_app(settings=settings, storage=InMemoryStorage()))


def test_health_shape() -> None:
    response = make_client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "db" in body
    assert "blob" in body


def test_models_lists_configured() -> None:
    body = make_client().get("/v1/models").json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2


def test_chat_completions_honest_501() -> None:
    response = make_client().post(
        "/v1/chat/completions",
        json={"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 501


def test_upload_rejects_unsupported_type() -> None:
    response = make_client().post(
        "/v1/kb/upload",
        files={"file": ("evil.exe", b"MZ", "application/x-msdownload")},
    )
    assert response.status_code == 415


def test_upload_rejects_oversize() -> None:
    response = make_client().post(
        "/v1/kb/upload",
        files={"file": ("big.txt", b"x" * 1025, "text/plain")},
    )
    assert response.status_code == 413


def test_upload_rejects_empty() -> None:
    response = make_client().post(
        "/v1/kb/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422


def test_upload_roundtrip_text() -> None:
    response = make_client().post(
        "/v1/kb/upload",
        files={"file": ("note.txt", b"hello rag", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["bytes"] == 9
    assert body["blob_url"].startswith("memory://")


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
        azurite_connection_string=(
            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
            "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq"
            "/K1SZFPTOtr/KBHBeksoGMGw==;"
            "BlobEndpoint=http://127.0.0.1:19999/devstoreaccount1;"
        ),
    )
    client = TestClient(create_app(settings=settings, storage=None))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["blob"] is False
