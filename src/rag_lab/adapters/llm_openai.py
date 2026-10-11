"""OpenAI-compatible LLM/embedding transport.

One client, many backends: Ollama locally (default) or any provider
by swapping base_url/key/model. No framework logic here — the
PydanticAI generation layer calls through this port in Phase 3.
"""

from typing import Protocol

from openai import OpenAI


class LlmPort(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


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

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
