"""Database helpers: engine, schema creation, idempotent upserts, price loading."""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import ROOT, get_secret

SCHEMA_FILE = ROOT / "sql" / "schema.sql"


def get_engine() -> Engine:
    url = get_secret("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Set it as an environment variable or in "
            ".streamlit/secrets.toml (see .streamlit/secrets.toml.example)."
        )
    return create_engine(url, pool_pre_ping=True)


def init_schema(engine: Engine) -> None:
    """Create all tables if they do not exist yet (idempotent)."""
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_FILE.read_text())


def _to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of plain-Python dicts (numpy scalars and NaN break psycopg2)."""
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict("records")


def upsert_df(engine: Engine, table: str, df: pd.DataFrame,
              key_cols: list[str], update: bool = True) -> None:
    """Insert rows; on a primary-key clash either update the row or skip it.

    This is what makes the pipeline safe to run repeatedly: re-running never
    creates duplicates. `update=True` refreshes existing rows (needed for
    daily prices, because Yahoo restates split/dividend-adjusted history).
    `update=False` keeps the first value (fine for immutable 1-minute bars).
    `table` and column names come from our own code, never from user input.
    """
    if df.empty:
        return
    cols = list(df.columns)
    insert = (f"INSERT INTO {table} ({', '.join(cols)}) "
              f"VALUES ({', '.join(':' + c for c in cols)})")
    value_cols = [c for c in cols if c not in key_cols]
    if update and value_cols:
        sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in value_cols)
        insert += f" ON CONFLICT ({', '.join(key_cols)}) DO UPDATE SET {sets}"
    else:
        insert += f" ON CONFLICT ({', '.join(key_cols)}) DO NOTHING"
    with engine.begin() as conn:
        conn.execute(text(insert), _to_records(df))


def count_rows(engine: Engine, table: str, symbol: str) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT COUNT(*) FROM {table} WHERE symbol = :s"), {"s": symbol}
        ).scalar_one()


def load_daily_prices(engine: Engine, symbol: str) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT date, open, high, low, close, volume FROM daily_prices "
             "WHERE symbol = :s ORDER BY date"),
        engine, params={"s": symbol},
    )
    df["date"] = pd.to_datetime(df["date"])
    return df
