"""Simple, explainable risk indicators computed from daily closes (backward-looking only)."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualised_volatility(returns: pd.Series, window: int = 20) -> float:
    """Latest rolling std of daily returns, scaled by sqrt(252)."""
    r = returns.dropna()
    if len(r) < window:
        return float("nan")
    return float(r.tail(window).std() * np.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series) -> float:
    """Worst peak-to-trough fall over the period, as a negative fraction (e.g. -0.35)."""
    c = close.dropna()
    if c.empty:
        return float("nan")
    return float((c / c.cummax() - 1).min())


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """1-day historical Value-at-Risk: the loss exceeded on only (1-level) of past days.

    Returned as a positive number (0.03 = a 3% one-day loss). It describes the past
    distribution; it is not a forecast and says nothing about losses beyond that threshold.
    """
    r = returns.dropna()
    if len(r) < 50:
        return float("nan")
    return float(-np.percentile(r, (1 - level) * 100))
