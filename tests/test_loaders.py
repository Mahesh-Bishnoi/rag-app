from rag_lab.ingest.loaders import TextLoader


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
