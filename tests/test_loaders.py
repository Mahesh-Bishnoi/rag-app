from conftest import encrypted_pdf, minimal_pdf

from rag_lab.ingest.loaders import PdfLoader, TextLoader


def test_text_loader_accepts_txt_and_markdown() -> None:
    loader = TextLoader()
    assert loader.can_handle("a.txt", "text/plain")
    assert loader.can_handle("a.md", "text/markdown")
    assert not loader.can_handle("a.pdf", "application/pdf")
    assert not loader.can_handle("a.exe", "application/x-msdownload")


def test_text_loader_emits_chunk_with_offsets() -> None:
    chunks = TextLoader().load(b"hello", "notes/note.txt")
    assert len(chunks) == 1
    assert chunks[0].title == "note.txt"
    assert chunks[0].byte_start == 0
    assert chunks[0].byte_end == 5


def test_pdf_loader_accepts_pdf_only() -> None:
    loader = PdfLoader()
    assert loader.can_handle("doc.pdf", "application/pdf")
    assert loader.can_handle("DOC.PDF", "application/octet-stream")
    assert loader.can_handle("unknown.bin", "application/pdf")
    assert not loader.can_handle("note.txt", "text/plain")


def test_pdf_loader_emits_one_chunk_per_page() -> None:
    chunks = PdfLoader().load(minimal_pdf(["hello pdf", "second page"]), "kb/doc.pdf")
    assert [chunk.content for chunk in chunks] == ["hello pdf", "second page"]
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert all(chunk.title == "doc.pdf" for chunk in chunks)
    assert all(chunk.mime == "application/pdf" for chunk in chunks)


def test_pdf_loader_skips_empty_pages() -> None:
    chunks = PdfLoader().load(minimal_pdf(["", "real text"]), "doc.pdf")
    assert [chunk.content for chunk in chunks] == ["real text"]
    assert [chunk.page for chunk in chunks] == [2]


def test_pdf_loader_rejects_corrupt_bytes() -> None:
    try:
        PdfLoader().load(b"%PDF-1.4 fake", "doc.pdf")
    except ValueError:
        return
    raise AssertionError("expected ValueError for corrupt PDF")


def test_pdf_loader_rejects_encrypted_pdf() -> None:
    try:
        PdfLoader().load(encrypted_pdf(), "doc.pdf")
    except ValueError:
        return
    raise AssertionError("expected ValueError for encrypted PDF")
