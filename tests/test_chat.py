"""Chat completions: OpenAI shape + citations over retrieve → rerank → generate."""

from typing import cast

from fastapi.testclient import TestClient

from rag_lab.adapters.documents import InMemoryDocumentStore
from rag_lab.adapters.embeddings import EmbeddingPort, StubEmbedding
from rag_lab.adapters.llm_openai import LlmPort, StubLlm
from rag_lab.adapters.retrieval import InMemoryRetrieval
from rag_lab.adapters.storage import InMemoryStorage
from rag_lab.adapters.traces import InMemoryTraceStore
from rag_lab.api.main import create_app
from rag_lab.core.config import Settings
from rag_lab.core.models import Chunk
from rag_lab.ingest.loaders import PdfLoader, TextLoader
from rag_lab.worker.pipeline import drain


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://rag:rag@localhost:5432/raglab",
        azurite_connection_string="",
        max_upload_bytes=1024,
    )


def _seed() -> InMemoryRetrieval:
    return InMemoryRetrieval(
        [
            Chunk(
                id="chunk-cat",
                content="the cat sits on the mat",
                title="cat.txt",
                embedding=[1.0, 0.0, 0.0],
            ),
            Chunk(
                id="chunk-quantum",
                content="intro to quantum entanglement for beginners",
                title="quantum.txt",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )


def make_client(
    retriever: InMemoryRetrieval | None = None,
    llm: LlmPort | None = None,
    embedder: EmbeddingPort | None = None,
) -> tuple[TestClient, StubLlm]:
    stub = llm if isinstance(llm, StubLlm) else StubLlm("cited stub answer")
    client = TestClient(
        create_app(
            settings=_settings(),
            storage=InMemoryStorage(),
            doc_store=InMemoryDocumentStore(),
            embedder=embedder or StubEmbedding(dim=3),
            retriever=retriever if retriever is not None else InMemoryRetrieval(),
            llm=llm or stub,
            trace_store=InMemoryTraceStore(),
        )
    )
    return client, stub


def ask(client: TestClient, query: str, model: str = "qwen2.5:3b") -> dict[str, object]:
    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": query}]},
    )
    assert response.status_code == 200
    body: dict[str, object] = response.json()
    return body


def choice_list(body: dict[str, object]) -> list[dict[str, object]]:
    choices = body["choices"]
    assert isinstance(choices, list)
    return cast("list[dict[str, object]]", choices)


def citation_list(body: dict[str, object]) -> list[dict[str, object]]:
    citations = body["citations"]
    assert isinstance(citations, list)
    return cast("list[dict[str, object]]", citations)


def test_chat_returns_openai_shape_with_citations() -> None:
    client, stub = make_client(_seed())
    body = ask(client, "what is quantum entanglement?")
    assert body["object"] == "chat.completion"
    assert body["model"] == "qwen2.5:3b"
    assert str(body["id"]).startswith("chatcmpl-")
    choices = choice_list(body)
    assert len(choices) == 1
    assert choices[0]["message"] == {"role": "assistant", "content": "cited stub answer"}
    assert choices[0]["finish_reason"] == "stop"
    citations = citation_list(body)
    assert len(citations) == 2
    assert citations[0]["title"] == "quantum.txt"
    assert citations[0]["chunk_id"] == "chunk-quantum"
    prompt = stub.seen[0][1]["content"]
    assert "[1]" in prompt
    assert "quantum entanglement" in prompt


def test_chat_empty_kb_answers_without_citations() -> None:
    client, _ = make_client()
    body = ask(client, "anything?")
    message = cast("dict[str, object]", choice_list(body)[0]["message"])
    assert message["content"] == "cited stub answer"
    assert citation_list(body) == []


def test_chat_rejects_missing_or_blank_user_message() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "system", "content": "hi"}]},
    )
    assert response.status_code == 422
    blank = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "  "}]},
    )
    assert blank.status_code == 422


class _FailingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedder down")


class _FailingLlm:
    def chat(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("llm down")


def test_chat_embed_failure_is_500() -> None:
    client, _ = make_client(_seed(), embedder=_FailingEmbedder())
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500


def test_chat_llm_failure_is_500() -> None:
    client, _ = make_client(_seed(), llm=_FailingLlm())
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500


def test_chat_full_ingest_to_answer_flow() -> None:
    settings = _settings()
    docs = InMemoryDocumentStore()
    storage = InMemoryStorage()
    retriever = InMemoryRetrieval()
    client = TestClient(
        create_app(
            settings=settings,
            storage=storage,
            doc_store=docs,
            embedder=StubEmbedding(dim=3, value=0.5),
            retriever=retriever,
            llm=StubLlm("flow answer"),
            trace_store=InMemoryTraceStore(),
        )
    )
    initiated = client.post(
        "/v1/kb/uploads/initiate",
        json={"filename": "note.txt", "content_type": "text/plain", "size_bytes": 9},
    ).json()
    document_id = str(initiated["document_id"])
    name = str(initiated["upload_url"]).split("?", 1)[0].rsplit("/", 1)[-1]
    storage.put(name, b"hello rag", "text/plain")
    assert client.post(f"/v1/kb/uploads/{document_id}/complete").json()["status"] == "uploaded"
    assert drain(docs, storage, StubEmbedding(dim=3, value=0.5), [TextLoader(), PdfLoader()]) == 1
    for document_id in docs.document_ids():
        retriever.add(docs.chunks_for(document_id))

    body = ask(client, "hello?")
    message = cast("dict[str, object]", choice_list(body)[0]["message"])
    assert message["content"] == "flow answer"
    citations = citation_list(body)
    assert len(citations) == 1
    assert citations[0]["chunk_id"]
