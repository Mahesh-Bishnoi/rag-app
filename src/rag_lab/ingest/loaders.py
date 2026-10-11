"""Loader port: detect → extract (text + structure) → emit chunks."""

from io import BytesIO
from typing import Protocol

from pypdf import PdfReader

from rag_lab.core.models import Chunk


class LoaderPort(Protocol):
    mime: str

    def can_handle(self, filename: str, content_type: str) -> bool: ...
    def load(self, data: bytes, source: str) -> list[Chunk]: ...


class TextLoader:
    mime = "text/plain"

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.endswith((".txt", ".md")) or content_type.startswith("text/")

    def load(self, data: bytes, source: str) -> list[Chunk]:
        text = data.decode("utf-8", errors="strict")
        title = source.rsplit("/", 1)[-1]
        return [
            Chunk(
                content=text,
                title=title,
                heading_path="",
                mime=self.mime,
                byte_start=0,
                byte_end=len(data),
            )
        ]


class PdfLoader:
    """Text-only PDF extraction, one chunk per non-empty page."""

    mime = "application/pdf"

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".pdf") or content_type == "application/pdf"

    def load(self, data: bytes, source: str) -> list[Chunk]:
        try:
            reader = PdfReader(BytesIO(data))
        except Exception as e:
            raise ValueError(f"unreadable PDF from {source}") from e
        if reader.is_encrypted:
            raise ValueError(f"encrypted PDF from {source}")
        title = source.rsplit("/", 1)[-1]
        chunks: list[Chunk] = []
        for index, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as e:
                raise ValueError(f"unreadable PDF page from {source}") from e
            if text:
                chunks.append(
                    Chunk(
                        content=text,
                        title=title,
                        heading_path="",
                        page=index + 1,
                        mime=self.mime,
                    )
                )
        return chunks
