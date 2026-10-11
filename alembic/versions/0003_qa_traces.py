"""QA trace table for persisted per-question records (ADR-0015)."""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE qa_traces (
            trace_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            retrieved JSONB NOT NULL DEFAULT '[]',
            rerank_order JSONB NOT NULL DEFAULT '[]',
            prompt TEXT NOT NULL DEFAULT '',
            answer TEXT NOT NULL DEFAULT '',
            citations JSONB NOT NULL DEFAULT '[]',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS qa_traces")
