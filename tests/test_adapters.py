import os

import pytest

from rag_lab.adapters.db import db_ping, get_engine, sqlalchemy_url
from rag_lab.adapters.storage import AzuriteBlobStorage, InMemoryStorage
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
    assert store.blobs["a.txt"] == b"data"


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
    assert url.endswith(f"/{container}/probe.txt")
    downloaded = second._client.get_blob_client("probe.txt").download_blob().readall()
    assert downloaded == b"probe-data"
    second._client.delete_container()
