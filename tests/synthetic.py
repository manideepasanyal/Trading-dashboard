"""SYNTHETIC price data for testing code mechanics only.

These are random walks, not market data. Never report anything computed from
them as a result: they exist so tests can run without network access.
"""
import numpy as np
import pandas as pd


def make_prices(n: int = 400, seed: int = 0, start: str = "2021-01-04") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def as_yfinance_raw(prices: pd.DataFrame, symbol: str = "TEST") -> pd.DataFrame:
    """Mimic yfinance's MultiIndex-column output so clean_prices() is tested realistically."""
    raw = prices.set_index("date").rename(columns=str.capitalize)
    raw.index.name = "Date"
    raw.columns = pd.MultiIndex.from_product([raw.columns, [symbol]], names=["Price", "Ticker"])
    return raw
