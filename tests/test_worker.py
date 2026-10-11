"""Worker pipeline contract: claim → chunk → embed → finish, typed failures."""

from conftest import minimal_pdf

from rag_lab.adapters.documents import InMemoryDocumentStore
from rag_lab.adapters.embeddings import StubEmbedding
from rag_lab.adapters.storage import InMemoryStorage, blob_name
from rag_lab.core.models import Chunk, Document
from rag_lab.ingest.loaders import LoaderPort, PdfLoader, TextLoader
from rag_lab.worker.pipeline import drain, process_one

LOADERS: list[LoaderPort] = [TextLoader(), PdfLoader()]


def _stage(
    data: bytes, filename: str, mime: str
) -> tuple[InMemoryDocumentStore, InMemoryStorage, str]:
    storage = InMemoryStorage()
    docs = InMemoryDocumentStore()
    name = f"key-{filename}"
    storage.put(name, data, mime)
    document_id = docs.initiate(
        Document(source=filename, title=filename, mime=mime, blob_url=storage.blob_url(name))
    )
    uploaded = docs.mark_uploaded(document_id)
    assert uploaded is not None
    assert uploaded.status == "uploaded"
    return docs, storage, document_id


def test_blob_name_strips_scheme_and_query() -> None:
    assert blob_name("memory://abc-note.txt") == "abc-note.txt"
    assert blob_name("memory://abc-note.txt?token=x&expires_in=900") == "abc-note.txt"
    assert blob_name("http://127.0.0.1:10000/devstoreaccount1/kb-raw/abc-note.txt") == (
        "abc-note.txt"
    )


def test_drain_finishes_text_with_embeddings() -> None:
    docs, storage, document_id = _stage(b"hello rag", "note.txt", "text/plain")
    assert drain(docs, storage, StubEmbedding(), LOADERS) == 1
    view = docs.get_status(document_id)
    assert view is not None
    assert view.status == "finished"
    assert view.chunk_count == 1
    assert docs.chunks_for(document_id)[0].embedding == [0.0] * 768


def test_drain_splits_markdown_and_keeps_heading_path() -> None:
    body = b"# Alpha\n\nbody one\n\n## Beta\n\nbody two\n"
    docs, storage, document_id = _stage(body, "note.md", "text/markdown")
    assert drain(docs, storage, StubEmbedding(), LOADERS) == 1
    view = docs.get_status(document_id)
    assert view is not None
    assert view.status == "finished"
    assert view.chunk_count == 2
    assert [chunk.heading_path for chunk in docs.chunks_for(document_id)] == [
        "Alpha",
        "Alpha > Beta",
    ]


def test_drain_ingests_pdf_pages() -> None:
    docs, storage, document_id = _stage(
        minimal_pdf(["hello pdf", "page two"]), "doc.pdf", "application/pdf"
    )
    assert drain(docs, storage, StubEmbedding(), LOADERS) == 1
    view = docs.get_status(document_id)
    assert view is not None
    assert view.status == "finished"
    assert view.chunk_count == 2
    assert [chunk.page for chunk in docs.chunks_for(document_id)] == [1, 2]


def test_drain_is_idle_without_uploaded_rows() -> None:
    docs = InMemoryDocumentStore()
    storage = InMemoryStorage()
    assert process_one(docs, storage, StubEmbedding(), LOADERS) is None
    assert drain(docs, storage, StubEmbedding(), LOADERS) == 0


def test_missing_blob_fails_with_blob_missing() -> None:
    docs = InMemoryDocumentStore()
    document_id = docs.initiate(
        Document(source="note.txt", title="note.txt", mime="text/plain", blob_url="memory://gone")
    )
    docs.mark_uploaded(document_id)
    assert process_one(docs, InMemoryStorage(), StubEmbedding(), LOADERS) == document_id
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "blob_missing")


def test_corrupt_pdf_fails_with_decode_failed() -> None:
    docs, storage, document_id = _stage(b"%PDF-1.4 fake", "doc.pdf", "application/pdf")
    assert drain(docs, storage, StubEmbedding(), LOADERS) == 1
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "decode_failed")


def test_heading_only_markdown_fails_with_empty_content() -> None:
    docs, storage, document_id = _stage(b"# Only heading\n\n", "note.md", "text/markdown")
    assert drain(docs, storage, StubEmbedding(), LOADERS) == 1
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "empty_content")


def test_unhandled_mime_fails_with_type_mismatch() -> None:
    docs, storage, document_id = _stage(b"\x00\x01", "blob.bin", "application/octet-stream")
    assert drain(docs, storage, StubEmbedding(), LOADERS) == 1
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "type_mismatch")


class _FailingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("llm down")


def test_embed_failure_fails_with_embed_failed() -> None:
    docs, storage, document_id = _stage(b"hello rag", "note.txt", "text/plain")
    assert drain(docs, storage, _FailingEmbedder(), LOADERS) == 1
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "embed_failed")


class _FailingStore(InMemoryDocumentStore):
    def finish(self, document_id: str, chunks: list[Chunk]) -> None:
        raise RuntimeError("db down")


def test_finish_failure_fails_with_persist_failed() -> None:
    docs: InMemoryDocumentStore = _FailingStore()
    storage = InMemoryStorage()
    storage.put("key-note.txt", b"hello rag", "text/plain")
    document_id = docs.initiate(
        Document(
            source="note.txt",
            title="note.txt",
            mime="text/plain",
            blob_url=storage.blob_url("key-note.txt"),
        )
    )
    docs.mark_uploaded(document_id)
    assert drain(docs, storage, StubEmbedding(), LOADERS) == 1
    view = docs.get_status(document_id)
    assert view is not None
    assert (view.status, view.error_code) == ("failed", "persist_failed")
