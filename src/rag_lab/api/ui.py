"""Minimal knowledge-base page. One static file, no build step (ADR-0017)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

INDEX_PATH = Path(__file__).parent / "static" / "index.html"
_INDEX_HTML = INDEX_PATH.read_text(encoding="utf-8")


def register_ui(app: FastAPI) -> None:
    """Serve the drag-drop + chat page at GET /. API routes match first."""

    @app.get("/", include_in_schema=False)
    def ui_root() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)
