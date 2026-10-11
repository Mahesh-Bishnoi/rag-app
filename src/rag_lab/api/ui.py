"""Minimal knowledge-base page. One static file, no build step (ADR-0017)."""

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

INDEX_PATH = Path(__file__).parent / "static" / "index.html"


@lru_cache(maxsize=1)
def _index_html() -> str:
    """Read the page on first request, not at import, so imports stay side-effect free."""
    return INDEX_PATH.read_text(encoding="utf-8")


def register_ui(app: FastAPI) -> None:
    """Serve the drag-drop + chat page at GET /. API routes match first."""

    @app.get("/", include_in_schema=False)
    def ui_root() -> HTMLResponse:
        return HTMLResponse(_index_html())
