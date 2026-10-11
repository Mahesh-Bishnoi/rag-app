"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from collections.abc import Sequence

from alembic import op

revision: str = ${up_revision!r}
down_revision: str | Sequence[str] | None = ${down_revision!r}
branch_labels: str | Sequence[str] | None = ${branch_labels!r}
depends_on: str | Sequence[str] | None = ${depends_on!r}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
