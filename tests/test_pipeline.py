"""Pipeline tests. The database test needs a reachable Postgres via DATABASE_URL and is skipped otherwise."""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src import data_pipeline
from src.config import get_secret
from src.data_pipeline import clean_prices
from tests.synthetic import as_yfinance_raw, make_prices


def test_clean_prices_handles_multiindex_and_drops_bad_rows():
    prices = make_prices(50)
    raw = as_yfinance_raw(prices)
    raw.iloc[3, 0] = np.nan                                   # a NaN price
    dup = raw.iloc[[10]]                                       # duplicate date
    raw = pd.concat([raw, dup]).sort_index()
    raw.iloc[20, raw.columns.get_loc(("High", "TEST"))] = 1.0  # high < low -> invalid
    clean, report = clean_prices(raw, "TEST")
    assert report["dropped_nan"] == 1
    assert report["dropped_duplicate_dates"] == 1
    assert report["dropped_invalid"] == 1
    assert list(clean.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert clean["date"].is_monotonic_increasing and clean["date"].is_unique
    assert (clean["high"] >= clean["low"]).all()


def test_clean_prices_empty_raises():
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame(), "TEST")


@pytest.mark.skipif(not get_secret("DATABASE_URL"), reason="DATABASE_URL not set")
def test_pipeline_is_idempotent(monkeypatch):
    from src.db import get_engine, load_daily_prices
    engine = get_engine()
    synthetic = make_prices(120, seed=1)
    monkeypatch.setattr(data_pipeline, "download_daily",
                        lambda symbol, start, end: as_yfinance_raw(synthetic, symbol))
    sym = "ZZTEST"
    try:
        first = data_pipeline.run([sym], years=1, engine=engine)
        second = data_pipeline.run([sym], years=1, engine=engine)   # run again: no new rows
        assert first[sym].startswith("ok: 120 rows fetched, 120 new")
        assert second[sym].startswith("ok: 120 rows fetched, 0 new")
        assert len(load_daily_prices(engine, sym)) == 120
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM daily_prices WHERE symbol = :s"), {"s": sym})
