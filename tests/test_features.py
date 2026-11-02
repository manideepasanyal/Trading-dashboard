import numpy as np
import pandas as pd

from src.features import (FEATURE_COLUMNS, TARGET_RETURN, TARGET_UP, add_features,
                          split_labelled_and_latest)
from tests.synthetic import make_prices


def test_no_lookahead_features_identical_on_truncated_history():
    """Core leakage test: features at day t must not change when future data is removed."""
    prices = make_prices(300)
    full = add_features(prices)
    for t in (80, 150, 250):
        truncated = add_features(prices.iloc[: t + 1])
        pd.testing.assert_series_equal(
            full.loc[t, FEATURE_COLUMNS].astype(float),
            truncated.loc[t, FEATURE_COLUMNS].astype(float),
            check_names=False,
        )


def test_target_is_next_day_return_and_only_forward_looking_column():
    prices = make_prices(100)
    df = add_features(prices)
    expected = prices["close"].shift(-1) / prices["close"] - 1
    np.testing.assert_allclose(df[TARGET_RETURN].iloc[:-1], expected.iloc[:-1])
    assert df[TARGET_RETURN].iloc[-1] != df[TARGET_RETURN].iloc[-1]          # NaN: future unknown
    assert TARGET_RETURN not in FEATURE_COLUMNS and TARGET_UP not in FEATURE_COLUMNS
    up = df[TARGET_UP].iloc[:-1]
    assert set(up.unique()) <= {0.0, 1.0}
    assert ((df[TARGET_RETURN].iloc[:-1] > 0).astype(float) == up).all()


def test_changing_the_future_does_not_change_todays_features():
    prices = make_prices(200)
    base = add_features(prices)
    altered = prices.copy()
    altered.loc[151:, ["open", "high", "low", "close"]] *= 3      # wreck everything after day 150
    changed = add_features(altered)
    pd.testing.assert_series_equal(
        base.loc[150, FEATURE_COLUMNS].astype(float),
        changed.loc[150, FEATURE_COLUMNS].astype(float),
        check_names=False,
    )


def test_rsi_bounds_and_split():
    df = add_features(make_prices(400))
    rsi = df["rsi_14"].dropna()
    assert rsi.between(0, 100).all()
    labelled, latest = split_labelled_and_latest(df)
    assert labelled[FEATURE_COLUMNS].notna().all().all()
    assert labelled[TARGET_RETURN].notna().all()
    assert len(latest) == 1 and latest[TARGET_RETURN].isna().all()
    assert latest["date"].iloc[0] > labelled["date"].iloc[-1]


def test_known_ma_ratio_value():
    prices = make_prices(60)
    df = add_features(prices)
    manual = prices["close"].iloc[-5:].mean()
    assert abs(df["ma_5"].iloc[-1] - manual) < 1e-9
    assert abs(df["close_ma5_ratio"].iloc[-1] - (prices["close"].iloc[-1] / manual - 1)) < 1e-12
