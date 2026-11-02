"""Historical daily OHLCV pipeline: yfinance -> clean/validate -> PostgreSQL.

Run:  python -m src.data_pipeline            (5 years, all four tickers)
      python -m src.data_pipeline --years 8 --symbols AAPL MSFT

Safe to run repeatedly: rows are upserted on (symbol, date), so no duplicates.
"""
import argparse
import logging
import sys
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .config import SYMBOLS
from .db import count_rows, get_engine, init_schema, upsert_df

log = logging.getLogger("data_pipeline")

PRICE_COLS = ["open", "high", "low", "close"]
# A single-day move this large in *adjusted* data is suspicious (bad tick?), not impossible.
SUSPICIOUS_ABS_RETURN = 0.40


def download_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Download daily bars for ONE symbol (kept separate so tests can replace it).

    auto_adjust=True returns prices adjusted for splits and dividends, which is
    what you want for return calculations (a split must not look like a -50% day).
    """
    return yf.download(symbol, start=start.isoformat(), end=end.isoformat(),
                       interval="1d", auto_adjust=True, progress=False)


def clean_prices(raw: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    """Standardise and validate raw yfinance output. Returns (clean_df, report)."""
    if raw is None or raw.empty:
        raise ValueError(f"{symbol}: no data returned")

    df = raw.copy()
    # Recent yfinance versions return MultiIndex columns (field, ticker) even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"index": "date", "datetime": "date"})

    missing = {"date", *PRICE_COLS, "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {sorted(missing)}")
    df = df[["date", *PRICE_COLS, "volume"]]

    report = {"rows_raw": len(df)}
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    n = len(df)
    df = df.dropna()
    report["dropped_nan"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates(subset="date", keep="last")
    report["dropped_duplicate_dates"] = n - len(df)

    n = len(df)
    valid = ((df[PRICE_COLS] > 0).all(axis=1) & (df["high"] >= df["low"]) & (df["volume"] >= 0))
    df = df[valid]
    report["dropped_invalid"] = n - len(df)

    df = df.sort_values("date").reset_index(drop=True)
    df["volume"] = df["volume"].round().astype("int64")
    df["date"] = df["date"].dt.date

    big_moves = int((df["close"].pct_change().abs() > SUSPICIOUS_ABS_RETURN).sum())
    report["suspicious_moves"] = big_moves
    if big_moves:
        log.warning("%s: %d daily moves larger than %.0f%% - check the source data",
                    symbol, big_moves, SUSPICIOUS_ABS_RETURN * 100)

    if df.empty:
        raise ValueError(f"{symbol}: no valid rows after cleaning")
    report["rows_clean"] = len(df)
    return df, report


def run(symbols: list[str], years: int, engine=None) -> dict[str, str]:
    """Fetch, clean and store history. Returns {symbol: status} for reporting."""
    engine = engine or get_engine()
    init_schema(engine)
    end = date.today() + timedelta(days=1)   # yfinance's `end` is exclusive
    start = date.today() - timedelta(days=365 * years)

    status: dict[str, str] = {}
    for symbol in symbols:
        try:
            log.info("Downloading %s from %s ...", symbol, start)
            clean, report = clean_prices(download_daily(symbol, start, end), symbol)
            clean.insert(0, "symbol", symbol)
            before = count_rows(engine, "daily_prices", symbol)
            upsert_df(engine, "daily_prices", clean, key_cols=["symbol", "date"], update=True)
            after = count_rows(engine, "daily_prices", symbol)
            status[symbol] = (f"ok: {len(clean)} rows fetched, {after - before} new, "
                              f"{after} total (cleaning: {report})")
        except Exception as exc:                      # one bad ticker must not stop the others
            log.error("%s failed: %s", symbol, exc)
            status[symbol] = f"FAILED: {exc}"
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    status = run(args.symbols, args.years)
    for sym, msg in status.items():
        print(f"{sym}: {msg}")
    return 1 if all(s.startswith("FAILED") for s in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
