"""LSTM tests. Random-walk data is SYNTHETIC: these check mechanics, not market performance."""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.config import get_secret
from src.features import FEATURE_COLUMNS, TARGET_RETURN, add_features, split_labelled_and_latest
from src.train_lstm import (SEQ_LEN, chronological_split, make_windows, predict_latest,
                            save_results, train_lstm)
from tests.synthetic import make_prices


def _data(n=900, seed=4):
    return split_labelled_and_latest(add_features(make_prices(n, seed=seed)))


def test_windows_end_on_the_target_row_and_are_ordered():
    vals = np.arange(100 * 2, dtype=float).reshape(100, 2)
    w = make_windows(vals, 30)
    assert w.shape == (71, 30, 2)
    assert (w[0] == vals[0:30]).all() and (w[-1] == vals[70:100]).all()
    # window k's last row is row k+29: the row whose target is "tomorrow's return"
    assert (w[5, -1] == vals[34]).all()


def test_split_is_chronological_and_complete():
    tr, va, te = chronological_split(1000)
    assert tr.start == 0 and tr.stop == va.start and va.stop == te.start and te.stop == 1000
    assert (tr.stop - tr.start, va.stop - va.start) == (700, 150)


def test_scaler_uses_train_rows_only():
    """Wreck every row after the training period: the fitted scaler must not move."""
    labelled, _ = _data()
    base = train_lstm(labelled, epochs=1)
    n_samples = len(labelled) - SEQ_LEN + 1
    train_last_row = chronological_split(n_samples)[0].stop + SEQ_LEN - 2
    altered = labelled.copy()
    altered.loc[train_last_row + 1:, FEATURE_COLUMNS] *= 1000
    changed = train_lstm(altered, epochs=1)
    np.testing.assert_allclose(base["mean"], changed["mean"])
    np.testing.assert_allclose(base["std"], changed["std"])
    assert base["y_scale"] == changed["y_scale"]


def test_noise_gives_no_edge_and_metrics_are_regression_metrics():
    labelled, _ = _data()
    res = train_lstm(labelled, epochs=15)
    t = res["metrics"]["test"]
    assert set(t["lstm"]) >= {"mae", "rmse", "directional_accuracy"}
    assert "f1" not in t["lstm"] and "precision" not in t["lstm"]    # classification metrics stay separate
    assert 0.35 <= t["lstm"]["directional_accuracy"] <= 0.65
    # On noise the LSTM must not beat "predict zero" by a meaningful margin.
    assert t["lstm"]["rmse"] >= 0.9 * t["baseline_zero"]["rmse"]
    assert len(res["test_actual"]) == len(res["test_pred"]) == int(t["lstm"]["n_test_days"])


def test_harness_detects_a_deliberately_leaky_feature():
    """Give the model tomorrow's return as an input: it must find it and beat the baseline clearly."""
    labelled, _ = _data()
    labelled = labelled.assign(leak=labelled[TARGET_RETURN])
    res = train_lstm(labelled, feature_columns=FEATURE_COLUMNS + ["leak"], epochs=60)
    t = res["metrics"]["test"]
    assert t["lstm"]["directional_accuracy"] > 0.85
    assert t["lstm"]["rmse"] < 0.6 * t["baseline_zero"]["rmse"]


def test_latest_forecast_is_a_finite_number():
    labelled, latest = _data()
    res = train_lstm(labelled, epochs=2)
    forecast = predict_latest(res, labelled, latest)
    assert forecast is not None and np.isfinite(forecast) and abs(forecast) < 0.5


@pytest.mark.skipif(not get_secret("DATABASE_URL"), reason="DATABASE_URL not set")
def test_save_results_roundtrip():
    from src.db import get_engine, init_schema
    engine = get_engine()
    init_schema(engine)
    labelled, latest = _data(n=500, seed=6)
    res = train_lstm(labelled, epochs=2)
    run_id, sym = "lstm_test_run", "ZZLSTM"
    try:
        save_results(engine, run_id, sym, res, labelled, latest)
        with engine.connect() as c:
            n_m = c.execute(text("SELECT COUNT(*) FROM model_results WHERE run_id=:r"), {"r": run_id}).scalar_one()
            n_p = c.execute(text("SELECT COUNT(*) FROM predictions WHERE run_id=:r"), {"r": run_id}).scalar_one()
            live = c.execute(text("SELECT actual_return, predicted_return FROM predictions "
                                  "WHERE run_id=:r ORDER BY date DESC LIMIT 1"), {"r": run_id}).one()
        assert n_m > 15 and n_p == len(res["test_pred"]) + 1
        assert live.actual_return is None and live.predicted_return is not None
    finally:
        with engine.begin() as c:
            for t in ("model_results", "predictions"):
                c.execute(text(f"DELETE FROM {t} WHERE run_id=:r"), {"r": run_id})
