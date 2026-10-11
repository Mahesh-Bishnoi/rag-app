"""OpenAI-compatible chat transport.

One client, many backends: Ollama locally (default) or any provider
by swapping base_url/key/model. Embeddings live behind the separate
EmbeddingPort in adapters/embeddings.py. No framework logic here — the
PydanticAI generation layer calls through this port in Phase 3.
"""

from typing import Protocol, runtime_checkable

from openai import OpenAI


@runtime_checkable
class LlmPort(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatibleLlm:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        content = response.choices[0].message.content
        return content if content is not None else ""


class StubLlm:
    """Canned-answer fake sharing the LlmPort interface. Hermetic tests only."""

    def __init__(self, answer: str = "stub answer") -> None:
        self._answer = answer
        self.seen: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.seen.append(messages)
        return self._answer
