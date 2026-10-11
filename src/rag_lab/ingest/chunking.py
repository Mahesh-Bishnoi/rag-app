"""Structure-aware splitting: headings → sections, long sections → overlap windows."""

from rag_lab.core.models import Chunk

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 200


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
    parts: list[str] = []
    for paragraph in _paragraphs(text, max_chars):
        parts.append(paragraph)
    glued = _pack(parts, max_chars)
    return _with_overlap(glued, max_chars, overlap)


def _paragraphs(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            parts.append(block)
            continue
        parts.extend(_sentences(block, max_chars))
    return parts


def _sentences(block: str, max_chars: int) -> list[str]:
    import re

    pieces = re.split(r"(?<=[.!?])\s+", block.strip())
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            out.extend(piece[i : i + max_chars] for i in range(0, len(piece), max_chars))
    return [part for part in out if part]


def _pack(parts: list[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}\n\n{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = part
    if current:
        packed.append(current)
    return packed


def _with_overlap(pieces: list[str], max_chars: int, overlap: int) -> list[str]:
    if overlap == 0 or len(pieces) < 2:
        return pieces
    out = [pieces[0]]
    for piece in pieces[1:]:
        room = max_chars - len(piece)
        take = min(overlap, room, len(out[-1]))
        out.append(f"{out[-1][-take:]}{piece}" if take else piece)
    return out
