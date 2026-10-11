"""Minimal drag-drop UI at GET / (ADR-0017)."""

from fastapi.testclient import TestClient

from rag_lab.adapters.documents import InMemoryDocumentStore
from rag_lab.adapters.storage import AzuriteBlobStorage, InMemoryStorage
from rag_lab.api.main import create_app
from rag_lab.core.config import Settings

# Deliberately fake: this client only ever hits a closed port (connection
# refused precedes auth), so no credential — real or well-known — belongs here.
DEAD_PORT_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=aGk=;"
    "BlobEndpoint=http://127.0.0.1:19999/devstoreaccount1;"
)


def make_client() -> TestClient:
    settings = Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
    )
    return TestClient(
        create_app(settings=settings, storage=InMemoryStorage(), doc_store=InMemoryDocumentStore())
    )


def test_ui_root_serves_upload_and_chat_page() -> None:
    body = make_client().get("/").text
    assert "kb-drop" in body
    assert "chat-form" in body
    assert "/v1/kb/uploads/initiate" in body
    assert "/v1/chat/completions" in body


def test_ui_renders_server_data_as_text() -> None:
    body = make_client().get("/").text
    assert "innerHTML" not in body
    assert "textContent" in body


def test_ui_root_is_html_and_leaves_api_404s_alone() -> None:
    client = make_client()
    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    assert client.get("/v1/nope").status_code == 404


def test_ui_put_sends_blob_type_header() -> None:
    """Azure Put Blob requires x-ms-blob-type; the browser must send it."""
    assert "x-ms-blob-type" in make_client().get("/").text


def test_lifespan_runs_with_memory_storage() -> None:
    with make_client() as client:
        assert client.get("/").status_code == 200


def test_lifespan_fails_open_when_blob_down() -> None:
    """CORS ensure must not stop the API from serving when Azurite is down."""
    settings = Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
    )
    store = AzuriteBlobStorage(DEAD_PORT_CONNECTION_STRING, "kb-probe")
    app = create_app(settings=settings, storage=store, doc_store=InMemoryDocumentStore())
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
