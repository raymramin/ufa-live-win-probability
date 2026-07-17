"""Optional DB helpers — uses DATABASE_URL or ShownSpace-style env vars."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_dotenv_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parents[2]
    for candidate in (
        here / ".env",
        Path.home() / "Desktop" / "shownspace_backend" / "backend" / ".env",
        Path.home() / "Desktop" / "shownspace_backend" / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def get_engine():
    load_dotenv_files()
    url = (os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL") or "").strip()
    if not url:
        # Compose from pieces if present
        host = os.getenv("DB_HOST") or os.getenv("PGHOST")
        user = os.getenv("DB_USER") or os.getenv("PGUSER")
        password = os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD")
        name = os.getenv("DB_NAME") or os.getenv("PGDATABASE")
        port = os.getenv("DB_PORT") or os.getenv("PGPORT") or "5432"
        if host and user and password and name:
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    if not url:
        raise RuntimeError(
            "No database URL. Set DATABASE_URL in .env (see .env.example)."
        )
    from sqlalchemy import create_engine

    return create_engine(url)


def read_table(table: str, where: str | None = None) -> pd.DataFrame:
    from sqlalchemy import text

    engine = get_engine()
    sql = f'SELECT * FROM "{table}"'
    if where:
        sql += f" WHERE {where}"
    return pd.read_sql(text(sql), engine)
