"""Human journeys: upload → ingest → chat. Read these first; they are the spec."""

from fastapi.testclient import TestClient

from rag_lab.adapters.documents import InMemoryDocumentStore
from rag_lab.adapters.embeddings import StubEmbedding
from rag_lab.adapters.llm_openai import StubLlm
from rag_lab.adapters.retrieval import InMemoryRetrieval
from rag_lab.adapters.storage import InMemoryStorage
from rag_lab.adapters.traces import InMemoryTraceStore
from rag_lab.api.main import create_app
from rag_lab.core.config import Settings
from rag_lab.ingest.loaders import PdfLoader, TextLoader
from rag_lab.worker.pipeline import drain


def _harness() -> tuple[TestClient, InMemoryDocumentStore, InMemoryStorage, InMemoryRetrieval]:
    docs = InMemoryDocumentStore()
    storage = InMemoryStorage()
    retriever = InMemoryRetrieval()
    client = TestClient(
        create_app(
            settings=Settings(
                database_url="postgresql://rag:rag@localhost:5432/raglab",
                azurite_connection_string="",
                max_upload_bytes=1024,
            ),
            storage=storage,
            doc_store=docs,
            embedder=StubEmbedding(dim=3, value=0.5),
            retriever=retriever,
            llm=StubLlm("grounded answer"),
            trace_store=InMemoryTraceStore(),
        )
    )
    return client, docs, storage, retriever


def _put_fake(storage: InMemoryStorage, upload_url: str, data: bytes, mime: str) -> None:
    name = upload_url.split("?", 1)[0].rsplit("/", 1)[-1]
    storage.put(name, data, mime)


def test_human_upload_journey_text_file_becomes_searchable() -> None:
    """Drag-drop a .txt: initiate → PUT bytes → complete → drain → finished."""
    client, docs, storage, _ = _harness()

    initiated = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": "note.txt", "content_type": "text/plain", "size_bytes": 9},
    ).json()
    document_id = str(initiated["document_id"])
    assert client.get(f"/v1/kb/documents/{document_id}").json()["status"] == "initiated"

    _put_fake(storage, str(initiated["upload_url"]), b"hello rag", "text/plain")
    assert client.post(f"/v1/kb/uploads/{document_id}/complete").json()["status"] == "uploaded"

    assert drain(docs, storage, StubEmbedding(dim=3, value=0.5), [TextLoader(), PdfLoader()]) == 1
    final = client.get(f"/v1/kb/documents/{document_id}").json()
    assert final["status"] == "finished"
    assert final["chunk_count"] == 1
    assert final["error_code"] is None


def test_human_chat_journey_empty_kb_answers_without_citations() -> None:
    """Ask with nothing ingested: 200 + empty citations (not 404)."""
    client, _, _, _ = _harness()
    body = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "anything?"}]},
    ).json()
    assert body["object"] == "chat.completion"
    assert body["citations"] == []
    assert body["choices"][0]["message"]["content"] == "grounded answer"


def test_human_chat_journey_ingested_doc_returns_citation() -> None:
    """Full loop: upload a note, drain it, ask about it, get a citation back."""
    client, docs, storage, retriever = _harness()

    initiated = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": "note.txt", "content_type": "text/plain", "size_bytes": 9},
    ).json()
    document_id = str(initiated["document_id"])
    _put_fake(storage, str(initiated["upload_url"]), b"hello rag", "text/plain")
    client.post(f"/v1/kb/uploads/{document_id}/complete")
    drain(docs, storage, StubEmbedding(dim=3, value=0.5), [TextLoader(), PdfLoader()])
    for doc_id in docs.document_ids():
        retriever.add(docs.chunks_for(doc_id))

    body = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "hello?"}]},
    ).json()
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"]
    assert str(body["id"]).startswith("chatcmpl-")
