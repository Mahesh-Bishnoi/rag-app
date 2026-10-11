"""Embedding transport behind a port. Production adapter speaks OpenAI REST."""

from typing import Protocol

from openai import OpenAI


class EmbeddingPort(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedding:
    """Embed via any OpenAI-compatible provider (Ollama by default)."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


class StubEmbedding:
    """Deterministic fake: constant vector per text. Hermetic tests only."""

    def __init__(self, dim: int = 768, value: float = 0.0) -> None:
        self._dim = dim
        self._value = value

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[self._value] * self._dim for _ in texts]
