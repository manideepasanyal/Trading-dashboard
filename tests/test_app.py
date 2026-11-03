"""Renders the Streamlit app headlessly against SYNTHETIC data and checks nothing crashes.

Needs a throwaway database in TEST_DATABASE_URL (never your real one): the test seeds and then
deletes rows for the symbol AAPL, because the dashboard's dropdown only offers the real tickers.
"""
import os

import pandas as pd
import pytest
import streamlit as st
from sqlalchemy import text
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.db import get_engine, init_schema, upsert_df
from src.features import add_features, split_labelled_and_latest
from src.train_lstm import save_results as save_lstm, train_lstm
from src.train_xgb import run_walk_forward, save_results as save_xgb
from tests.synthetic import make_prices

TEST_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture()
def seeded(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    engine = get_engine()
    init_schema(engine)
    _cleanup(engine)
    yield engine
    _cleanup(engine)


def _cleanup(engine):
    with engine.begin() as c:
        for t in ("daily_prices", "market_data", "predictions", "feature_importance", "model_results"):
            c.execute(text(f"DELETE FROM {t} WHERE symbol = 'AAPL'"))


def _run_app():
    st.cache_data.clear()                 # query results are cached; start each render fresh
    st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    return at


def test_app_renders_with_no_data(seeded):
    at = _run_app()
    assert not at.exception
    texts = " ".join(i.value for i in at.info)
    assert "No intraday data" in texts and "No daily history" in texts and "No trained models" in texts


def test_app_renders_full_dashboard_with_models(seeded):
    prices = make_prices(900, seed=21)
    prices["date"] = prices["date"].dt.date
    prices.insert(0, "symbol", "AAPL")
    upsert_df(seeded, "daily_prices", prices, ["symbol", "date"])
    daily = pd.read_sql(text("SELECT date, open, high, low, close, volume FROM daily_prices "
                             "WHERE symbol='AAPL' ORDER BY date"), seeded)
    daily["date"] = pd.to_datetime(daily["date"])
    labelled, latest = split_labelled_and_latest(add_features(daily))
    save_xgb(seeded, "xgb_apptest", "AAPL", run_walk_forward(labelled, 3), latest)
    save_lstm(seeded, "lstm_apptest", "AAPL", train_lstm(labelled, epochs=2), labelled, latest)

    at = _run_app()
    assert not at.exception, [e.value for e in at.exception]
    labels = {m.label for m in at.metric}
    assert {"Volatility (20d, annualised)", "Max drawdown (shown period)",
            "1-day 95% historical VaR", "RSI (14)",
            "XGBoost: probability next day is UP", "LSTM: forecast next-day return"} <= labels
    assert len(at.dataframe) >= 2                      # classification + regression tables
    headers = " ".join(s.value for s in at.subheader)
    assert "Model performance comparison" in headers and "feature importance" in headers.lower()
    assert any("not evidence of a profitable strategy" in w.value for w in at.warning)
