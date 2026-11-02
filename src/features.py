"""Feature engineering for next-day return prediction.

THE ONE RULE (no data leakage): every feature in row t is computed only from
prices/volumes up to and including the close of day t. The only thing that looks
forward is the *target* (`target_return`, the return from close t to close t+1),
and it is kept in separate columns that are never used as model inputs.

All rolling / ewm operations below are backward-looking. `tests/test_features.py`
proves this: features computed on data truncated at day t equal the features
for day t computed on the full history.
"""
import numpy as np
import pandas as pd

MA_WINDOWS = (5, 10, 20, 50)
RETURN_LAGS = (1, 2, 3, 5)
VOL_WINDOWS = (5, 20)
RSI_PERIOD = 14
TRADING_DAYS = 252

# Columns the models are allowed to see. Everything else (raw prices, raw MAs,
# targets) is for display or labelling only.
FEATURE_COLUMNS = (
    ["ret_1d"]
    + [f"ret_lag_{k}" for k in RETURN_LAGS]
    + [f"close_ma{w}_ratio" for w in MA_WINDOWS]
    + [f"vol_{w}d" for w in VOL_WINDOWS]
    + ["rsi_14", "macd_pct", "macd_signal_pct", "macd_hist_pct",
       "volume_chg_1d", "volume_ratio_20"]
)
TARGET_RETURN = "target_return"
TARGET_UP = "target_up"


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI (0-100). Uses exponential smoothing with alpha = 1/period."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0     # only gains -> RSI 100
    return out


def add_features(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: columns date, open, high, low, close, volume (one symbol, any order)."""
    df = prices.sort_values("date").reset_index(drop=True).copy()
    close, volume = df["close"], df["volume"].astype(float)

    df["ret_1d"] = close.pct_change()
    for k in RETURN_LAGS:
        df[f"ret_lag_{k}"] = df["ret_1d"].shift(k)

    for w in MA_WINDOWS:
        ma = close.rolling(w).mean()
        df[f"ma_{w}"] = ma                                   # raw level: for charts only
        df[f"close_ma{w}_ratio"] = close / ma - 1            # scale-free: used by models

    for w in VOL_WINDOWS:
        df[f"vol_{w}d"] = df["ret_1d"].rolling(w).std()      # daily std dev of returns
    df["vol_20d_annualised"] = df["vol_20d"] * np.sqrt(TRADING_DAYS)   # display/risk

    df["rsi_14"] = rsi(close)

    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    # Divide by price so the value is comparable across time and across stocks.
    df["macd_pct"] = macd / close
    df["macd_signal_pct"] = signal / close
    df["macd_hist_pct"] = (macd - signal) / close

    df["volume_chg_1d"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    df["volume_ratio_20"] = volume / volume.rolling(20).mean()

    # Targets: the ONLY forward-looking columns. NaN on the last row (the future is unknown).
    df[TARGET_RETURN] = close.shift(-1) / close - 1
    df[TARGET_UP] = (df[TARGET_RETURN] > 0).astype(float).where(df[TARGET_RETURN].notna())
    return df


def split_labelled_and_latest(featured: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labelled, latest).

    labelled: rows with complete features AND a known next-day target (for training/evaluation).
    latest:   the most recent row with complete features and no target yet (for live prediction).
    """
    complete = featured.dropna(subset=FEATURE_COLUMNS)
    labelled = complete.dropna(subset=[TARGET_RETURN]).reset_index(drop=True)
    latest = complete[complete[TARGET_RETURN].isna()].tail(1).reset_index(drop=True)
    return labelled, latest
