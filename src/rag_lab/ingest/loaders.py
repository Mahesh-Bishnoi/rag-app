"""Loader port: detect → extract (text + structure) → emit chunks."""

from typing import Protocol

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
