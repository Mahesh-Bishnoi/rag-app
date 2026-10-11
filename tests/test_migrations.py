"""Live migration tests. Skipped without RAGLAB_LIVE_DB=1 (needs postgres)."""

import os
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic.config import Config
from sqlalchemy import MetaData
from sqlmodel import SQLModel

from alembic import command
from rag_lab.adapters.db import get_engine
from rag_lab.adapters.tables import ChunkRow, DocumentRow  # noqa: F401
from rag_lab.core.config import Settings

LIVE = os.environ.get("RAGLAB_LIVE_DB") == "1"
needs_live = pytest.mark.skipif(not LIVE, reason="needs live postgres (RAGLAB_LIVE_DB=1)")

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_DB = "raglab_migrtest"


def _base_url() -> str:
    return Settings().database_url.rsplit("/", 1)[0]


def _admin_connect() -> psycopg.Connection[Any]:
    return psycopg.connect(f"{_base_url()}/raglab", autocommit=True)


def assert_migrated_schema_matches_models(database_url: str) -> None:
    engine = get_engine(database_url)
    try:
        reflected = MetaData()
        reflected.reflect(bind=engine)
        assert set(SQLModel.metadata.tables) <= set(reflected.tables)
        for name in SQLModel.metadata.tables:
            assert {c.name for c in SQLModel.metadata.tables[name].columns} == {
                c.name for c in reflected.tables[name].columns
            }
    finally:
        engine.dispose()


@needs_live
def test_migration_upgrade_downgrade_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _admin_connect() as conn, conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {SCRATCH_DB}")
        cur.execute(f"CREATE DATABASE {SCRATCH_DB}")
    try:
        monkeypatch.setenv("DATABASE_URL", f"{_base_url()}/{SCRATCH_DB}")
        cfg = Config(str(ROOT / "alembic.ini"))
        command.upgrade(cfg, "head")
        with psycopg.connect(f"{_base_url()}/{SCRATCH_DB}") as conn, conn.cursor() as cur:
            cur.execute("SELECT version_num FROM alembic_version")
            version = cur.fetchone()
            assert version is not None
            assert version[0] == "0002"
            cur.execute("SELECT to_regclass('public.documents'), to_regclass('public.chunks')")
            tables = cur.fetchone()
            assert tables is not None
            assert all(table is not None for table in tables)
        assert_migrated_schema_matches_models(f"{_base_url()}/{SCRATCH_DB}")
        command.downgrade(cfg, "base")
        with psycopg.connect(f"{_base_url()}/{SCRATCH_DB}") as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.documents'), to_regclass('public.chunks')")
            assert cur.fetchone() == (None, None)
    finally:
        with _admin_connect() as conn, conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {SCRATCH_DB}")
