"""Typed startup config. Loaded once at startup, passed in as values."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", frozen=True)

    database_url: str = "postgresql://rag:rag@localhost:5432/raglab"
    azurite_connection_string: str = ""
    blob_container: str = "kb-raw"
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    openai_model: str = "qwen2.5:3b"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    # DDL truth for chunks.embedding vector(n); the query path reads the
    # actual dim off the model output and rejects mismatches (ADR-0014).
    embedding_dimensions: int = Field(default=768, gt=0, le=4096)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    # Origins the dev-emulator CORS rule grants PUT access (ADR-0017). Only
    # ever written to devstoreaccount1; real Azure keeps operator-owned CORS.
    blob_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"]
    )
    ingest_poll_seconds: float = Field(default=1.0, gt=0)
    retrieval_candidates: int = Field(default=40, gt=0, le=200)
    answer_context_chunks: int = Field(default=5, gt=0, le=20)
