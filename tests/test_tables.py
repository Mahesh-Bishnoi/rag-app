"""Table-model contract: SQLModel metadata must mirror migration 0001."""

import uuid

from sqlmodel import SQLModel

from rag_lab.adapters.tables import ChunkRow, DocumentRow


def test_metadata_tables_match_baseline() -> None:
    assert set(SQLModel.metadata.tables) == {"documents", "chunks"}
    assert models_columns("documents") == {
        "id",
        "source",
        "title",
        "mime",
        "blob_url",
        "status",
        "error_code",
        "created_at",
    }
    assert models_columns("chunks") == {
        "id",
        "document_id",
        "content",
        "title",
        "heading_path",
        "page",
        "mime",
        "byte_start",
        "byte_end",
        "embedding",
        "text_search",
    }


def test_table_rows_validate_required_fields() -> None:
    ChunkRow(document_id=uuid.uuid4(), content="hello")
    DocumentRow(source="s", title="t", mime="text/plain", blob_url="memory://x")


def test_chunks_carry_hybrid_search_indexes() -> None:
    indexes = {index.name for index in SQLModel.metadata.tables["chunks"].indexes}
    assert {"chunks_text_search_idx", "chunks_embedding_hnsw_idx"} <= indexes


def models_columns(table: str) -> set[str]:
    return {c.name for c in SQLModel.metadata.tables[table].columns}
