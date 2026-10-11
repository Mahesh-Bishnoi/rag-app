"""EmbeddingPort contract: stub is deterministic, prod client builds without network."""

from rag_lab.adapters.embeddings import OpenAICompatibleEmbedding, StubEmbedding


def test_stub_returns_constant_vectors_per_text() -> None:
    vectors = StubEmbedding(dim=4, value=0.5).embed(["a", "bb"])
    assert vectors == [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]


def test_stub_default_dim_matches_chunk_column() -> None:
    assert len(StubEmbedding().embed(["x"])[0]) == 768


def test_openai_compatible_embedding_builds_without_network() -> None:
    embedder = OpenAICompatibleEmbedding(
        base_url="http://127.0.0.1:1/v1", api_key="test", model="nomic-embed-text"
    )
    assert embedder is not None
