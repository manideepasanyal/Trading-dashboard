"""Intraday (1-minute) collector feeding the dashboard's live price chart.

Run:  python etl_yfinance.py

Pulls the last trading day of 1-minute bars per ticker and stores them in
`market_data`. Re-running is safe: the table's primary key (symbol, datetime)
plus ON CONFLICT DO NOTHING means no duplicate bars. For multi-year *daily*
history used by the forecasting models, see `python -m src.data_pipeline`.
"""
import pandas as pd
import yfinance as yf

from src.config import SYMBOLS
from src.db import get_engine, init_schema, upsert_df


def fetch_intraday(symbol: str) -> pd.DataFrame:
    # One symbol per request keeps the column layout predictable.
    raw = yf.download([symbol], period="1d", interval="1m", auto_adjust=True, progress=False)
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    df = df.rename(columns={df.columns[0]: "datetime"})        # 'Datetime' (1m) or 'Date'
    df.columns = [str(c).lower() for c in df.columns]
    df = df[["datetime", "open", "high", "low", "close", "volume"]].dropna()
    df["volume"] = df["volume"].round().astype("int64")
    df.insert(0, "symbol", symbol)
    return df


def main() -> None:
    engine = get_engine()
    init_schema(engine)
    for symbol in SYMBOLS:
        print(f"Fetching {symbol}...")
        df = fetch_intraday(symbol)
        if df.empty:
            print(f"No data for {symbol}, skipping.")
            continue
        upsert_df(engine, "market_data", df, key_cols=["symbol", "datetime"], update=False)
        print(f"  stored up to {len(df)} bars (existing bars are skipped)")


if __name__ == "__main__":
    main()
