"""Baseline schema: extensions + documents + chunks.

Async ingest (ADR-0013): documents carry a stage status + typed error_code
with a partial index over claimable rows. Still in development with no prod
data, so the baseline is rewritten instead of appended. Existing dev volumes
must be recreated: compose down -v && compose up -d && alembic upgrade head.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        CREATE TABLE documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            mime TEXT NOT NULL,
            blob_url TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'initiated'
                CHECK (status IN (
                    'initiated', 'uploaded', 'chunking',
                    'embedding', 'finished', 'failed')),
            error_code TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            heading_path TEXT NOT NULL DEFAULT '',
            page INT,
            mime TEXT NOT NULL DEFAULT 'text/plain',
            byte_start INT,
            byte_end INT,
            embedding vector(768),
            text_search tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        )
        """
    )
    op.execute("CREATE INDEX chunks_document_id_idx ON chunks (document_id)")
    op.execute("CREATE INDEX chunks_text_search_idx ON chunks USING GIN (text_search)")
    op.execute(
        "CREATE INDEX documents_claim_idx ON documents (created_at) WHERE status = 'uploaded'"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
