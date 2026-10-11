"""Structure-aware splitting: headings → sections, long sections → overlap windows.

Heading detection stays custom (markdown `#` paths); the windowing step
delegates to LangChain's RecursiveCharacterTextSplitter (ADR-0016 seam:
LangChain owns splitters, configured here, never bypassed).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_lab.core.models import Chunk

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 200

_SEPARATORS = ["\n\n", "\n", " ", ""]


def split_chunks(
    chunks: list[Chunk],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split loader chunks into retrieval-sized pieces. Pure; no IO."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")
    out: list[Chunk] = []
    for chunk in chunks:
        sections = _sections(chunk)
        preserve_offsets = len(sections) == 1
        for heading_path, text in sections:
            pieces = _windows(text, max_chars, overlap)
            for piece in pieces:
                single = preserve_offsets and len(pieces) == 1 and piece == chunk.content
                out.append(
                    Chunk(
                        content=piece,
                        title=chunk.title,
                        heading_path=heading_path,
                        page=chunk.page,
                        mime=chunk.mime,
                        byte_start=chunk.byte_start if single else None,
                        byte_end=chunk.byte_end if single else None,
                    )
                )
    return out


def _sections(chunk: Chunk) -> list[tuple[str, str]]:
    """Split content on markdown headings; empty bodies are dropped."""
    stack: list[str] = []
    current: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        body = "".join(current).strip()
        if body:
            path = " > ".join(stack) if stack else chunk.heading_path
            sections.append((path, body))
        current.clear()

    for line in chunk.content.splitlines(keepends=True):
        level, title = _heading(line)
        if level == 0:
            current.append(line)
            continue
        flush()
        stack = [*stack[: level - 1], title]
    flush()
    if not sections and not stack:
        text = chunk.content.strip()
        return [(chunk.heading_path, text)] if text else []
    return sections


def _heading(line: str) -> tuple[int, str]:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return (0, "")
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes > 6:
        return (0, "")
    rest = stripped[hashes:]
    # Require "# title" with a space; "#tag" is body text.
    if not rest.startswith((" ", "\t")):
        return (0, "")
    title = rest.strip()
    return (hashes, title) if title else (0, "")


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=overlap, separators=_SEPARATORS
    )
    return [part for part in splitter.split_text(text) if part.strip()]
