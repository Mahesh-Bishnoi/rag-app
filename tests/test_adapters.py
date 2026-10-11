import os

import pytest

from rag_lab.adapters.db import db_ping, get_engine, sqlalchemy_url
from rag_lab.adapters.storage import AzuriteBlobStorage, BlobStat, InMemoryStorage
from rag_lab.core.config import Settings

LIVE = os.environ.get("RAGLAB_LIVE_DB") == "1"
needs_live = pytest.mark.skipif(not LIVE, reason="needs live services (RAGLAB_LIVE_DB=1)")

# Deliberately fake: this client only ever hits a closed port (connection
# refused precedes auth), so no credential — real or well-known — belongs here.
DEAD_PORT_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=aGk=;"
    "BlobEndpoint=http://127.0.0.1:19999/devstoreaccount1;"
)


def test_sqlalchemy_url_uses_psycopg_driver() -> None:
    url = sqlalchemy_url("postgresql://rag:rag@localhost:5432/raglab")
    assert url == "postgresql+psycopg://rag:rag@localhost:5432/raglab"
    engine = get_engine("postgresql://rag:rag@localhost:5432/raglab")
    assert engine.url.drivername == "postgresql+psycopg"
    engine.dispose()


def test_db_ping_false_on_refused_port() -> None:
    assert not db_ping("postgresql://rag:rag@127.0.0.1:1/raglab")


def test_in_memory_storage_roundtrip() -> None:
    store = InMemoryStorage()
    assert store.ping()
    url = store.put("a.txt", b"data", "text/plain")
    assert url == "memory://a.txt"
    assert store.blobs["a.txt"].data == b"data"
    assert store.stat("a.txt") == BlobStat(size=4, content_type="text/plain")
    assert store.download("a.txt") == b"data"
    assert store.stat("missing.txt") is None
    assert store.download("missing.txt") is None


def test_in_memory_issue_upload_url_mints_expiring_url() -> None:
    store = InMemoryStorage()
    url, expires_in = store.issue_upload_url("doc.pdf", "application/pdf")
    assert "doc.pdf" in url
    assert url != store.blob_url("doc.pdf")
    assert expires_in == 900
    assert store.blob_url("doc.pdf") == "memory://doc.pdf"


def test_azurite_ping_false_on_closed_port() -> None:
    store = AzuriteBlobStorage(DEAD_PORT_CONNECTION_STRING, "kb-probe")
    assert not store.ping()


@needs_live
def test_azurite_put_roundtrip() -> None:
    settings = Settings()
    assert settings.azurite_connection_string, "AZURITE_CONNECTION_STRING not configured"
    container = "kb-test-scratch"
    first = AzuriteBlobStorage(settings.azurite_connection_string, container)
    second = AzuriteBlobStorage(settings.azurite_connection_string, container)
    first.put("probe.txt", b"probe-data", "text/plain")
    second.put("probe.txt", b"probe-data", "text/plain")
    assert first.ping()
    url = second.put("probe.txt", b"probe-data", "text/plain")
    assert url.endswith(f"/{container}/probe.txt")
    downloaded = second._client.get_blob_client("probe.txt").download_blob().readall()
    assert downloaded == b"probe-data"
    second._client.delete_container()


@needs_live
def test_azurite_signed_url_put_then_stat_download() -> None:
    import httpx

    settings = Settings()
    assert settings.azurite_connection_string, "AZURITE_CONNECTION_STRING not configured"
    container = "kb-test-sas"
    store = AzuriteBlobStorage(settings.azurite_connection_string, container)
    try:
        upload_url, expires_in = store.issue_upload_url("sas-probe.txt", "text/plain")
        assert "sig=" in upload_url
        assert expires_in == 900
        response = httpx.put(
            upload_url,
            content=b"sas-data",
            headers={"Content-Type": "text/plain", "x-ms-blob-type": "BlockBlob"},
        )
        assert response.status_code in (200, 201)
        assert store.stat("sas-probe.txt") == BlobStat(size=8, content_type="text/plain")
        assert store.download("sas-probe.txt") == b"sas-data"
    finally:
        store._client.delete_container()
