"""Postgres helpers. Connection opened per check; pools come in Phase 3."""

import psycopg
from sqlalchemy import Engine, create_engine


def db_ping(database_url: str) -> bool:
    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() is not None
    except Exception:
        return False


def sqlalchemy_url(database_url: str) -> str:
    return database_url.replace("postgresql://", "postgresql+psycopg://")


def get_engine(database_url: str) -> Engine:
    return create_engine(sqlalchemy_url(database_url))
