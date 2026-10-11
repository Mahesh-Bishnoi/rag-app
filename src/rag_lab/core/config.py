"""Typed startup config. Loaded once at startup, passed in as values."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql://rag:rag@localhost:5432/raglab"
    azurite_connection_string: str = ""
    blob_container: str = "kb-raw"
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    openai_model: str = "qwen2.5:3b"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    max_upload_bytes: int = 20 * 1024 * 1024
