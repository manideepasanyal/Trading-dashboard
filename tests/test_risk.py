import numpy as np
import pandas as pd
import pytest

from src.risk import annualised_volatility, historical_var, max_drawdown


def test_max_drawdown_known_value():
    close = pd.Series([100, 120, 90, 110, 80, 100.0])
    assert max_drawdown(close) == pytest.approx(80 / 120 - 1)      # peak 120 -> trough 80


def test_volatility_scaling():
    r = pd.Series([0.01, -0.01] * 30)
    assert annualised_volatility(r, 20) == pytest.approx(r.tail(20).std() * np.sqrt(252))
    assert np.isnan(annualised_volatility(pd.Series([0.01] * 5), 20))


def test_var_known_value_and_sign():
    r = pd.Series(np.linspace(-0.05, 0.05, 101))        # evenly spread returns
    var = historical_var(r, 0.95)
    assert var == pytest.approx(0.045, abs=1e-9)          # 5th percentile is -0.045
    assert var > 0
    assert np.isnan(historical_var(pd.Series([0.01] * 10)))
