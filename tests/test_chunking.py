"""Structure-aware chunking contract: headings, pages, overlap."""

from rag_lab.core.models import Chunk
from rag_lab.ingest.chunking import split_chunks


def _chunk(content: str, **overrides: object) -> Chunk:
    base: dict[str, object] = {
        "content": content,
        "title": "note.md",
        "heading_path": "",
        "mime": "text/markdown",
    }
    base.update(overrides)
    return Chunk.model_validate(base)


def test_short_chunk_passes_through_with_offsets() -> None:
    chunk = _chunk("hello", byte_start=0, byte_end=5)
    out = split_chunks([chunk])
    assert len(out) == 1
    assert out[0].content == "hello"
    assert out[0].byte_start == 0
    assert out[0].byte_end == 5


def test_markdown_headings_become_heading_path() -> None:
    content = "# Alpha\n\nbody one\n\n## Beta\n\nbody two\n"
    out = split_chunks([_chunk(content)], max_chars=500, overlap=0)
    assert [chunk.heading_path for chunk in out] == ["Alpha", "Alpha > Beta"]
    assert out[0].title == "note.md"


def test_long_text_splits_on_paragraphs_with_overlap() -> None:
    para = "x" * 400
    content = f"{para}\n\n{para}\n\n{para}"
    out = split_chunks([_chunk(content)], max_chars=500, overlap=50)
    assert len(out) == 3
    assert all(len(chunk.content) <= 500 for chunk in out)
    assert out[1].content.startswith(out[0].content[-50:])
    assert all(chunk.content for chunk in out)


def test_pdf_pages_split_independently_and_keep_page() -> None:
    first = _chunk("a" * 900, mime="application/pdf", page=1)
    second = _chunk("b" * 900, mime="application/pdf", page=2)
    out = split_chunks([first, second], max_chars=500, overlap=0)
    assert [chunk.page for chunk in out] == [1, 1, 2, 2]
    assert all(chunk.mime == "application/pdf" for chunk in out)


def test_empty_pieces_never_emitted() -> None:
    out = split_chunks([_chunk("# Only heading\n\n")], max_chars=500, overlap=0)
    assert out == []
