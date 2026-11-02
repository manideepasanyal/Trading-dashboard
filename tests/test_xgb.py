"""XGBoost pipeline tests. Random-walk data is SYNTHETIC: these check mechanics, not market performance."""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.config import get_secret
from src.features import FEATURE_COLUMNS, TARGET_RETURN, add_features, split_labelled_and_latest
from src.metrics import classification_metrics, directional_accuracy, regression_metrics
from src.train_xgb import run_walk_forward, save_results, walk_forward_splits
from tests.synthetic import make_prices


def _labelled(n=1100, seed=3):
    labelled, latest = split_labelled_and_latest(add_features(make_prices(n, seed=seed)))
    return labelled, latest


def test_walk_forward_splits_chronological_expanding_and_disjoint():
    splits = list(walk_forward_splits(1000, n_folds=5))
    assert len(splits) == 5
    prev_end = None
    prev_train = 0
    for tr, te in splits:
        assert tr.max() < te.min()                      # train strictly before test
        assert len(tr) > prev_train                     # window expands
        if prev_end is not None:
            assert te.min() == prev_end                 # test blocks are consecutive, no overlap
        prev_end, prev_train = te.max() + 1, len(tr)
    assert prev_end == 1000                             # every row after the initial window is tested once


def test_metrics_known_values():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([1, 0, 0, 1, 1])        # TP=2, FN=1, TN=1, FP=1
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    r = regression_metrics(np.array([0.01, -0.02, 0.03]), np.array([0.02, -0.01, -0.01]))
    assert r["mae"] == pytest.approx((0.01 + 0.01 + 0.04) / 3)
    assert r["rmse"] == pytest.approx(np.sqrt((0.01**2 + 0.01**2 + 0.04**2) / 3))
    assert directional_accuracy([0.01, -0.02, 0.03], [0.02, -0.01, -0.01]) == pytest.approx(2 / 3)


def test_random_walk_gives_no_edge_so_no_leakage():
    """Prices here are pure noise, so honest out-of-sample accuracy must sit near 50%.
    A leaky pipeline would score far higher."""
    labelled, _ = _labelled()
    res = run_walk_forward(labelled, n_folds=5)
    acc = res["metrics"]["xgboost"]["accuracy"]
    assert 0.40 <= acc <= 0.60, acc
    assert res["metrics"]["xgboost"]["n_test_days"] > 400
    assert set(res["importance"].index) == set(FEATURE_COLUMNS)


def test_harness_detects_a_deliberately_leaky_feature():
    """Sensitivity check: give the model tomorrow's return as a feature; it must score ~100%."""
    labelled, _ = _labelled()
    labelled = labelled.assign(leak=labelled[TARGET_RETURN])
    res = run_walk_forward(labelled, n_folds=5, feature_columns=FEATURE_COLUMNS + ["leak"])
    assert res["metrics"]["xgboost"]["accuracy"] > 0.95


def test_predictions_cover_only_out_of_sample_dates():
    labelled, _ = _labelled()
    res = run_walk_forward(labelled, n_folds=5)
    first_test_date = labelled["date"].iloc[int(len(labelled) * 0.5)]
    assert pd.to_datetime(res["oof"]["date"]).min() == first_test_date


@pytest.mark.skipif(not get_secret("DATABASE_URL"), reason="DATABASE_URL not set")
def test_save_results_roundtrip():
    from src.db import get_engine, init_schema
    engine = get_engine()
    init_schema(engine)
    labelled, latest = _labelled(n=700, seed=5)
    res = run_walk_forward(labelled, n_folds=3)
    run_id, sym = "xgb_test_run", "ZZXGB"
    try:
        save_results(engine, run_id, sym, res, latest)
        with engine.connect() as c:
            n_metrics = c.execute(text("SELECT COUNT(*) FROM model_results WHERE run_id=:r"), {"r": run_id}).scalar_one()
            n_pred = c.execute(text("SELECT COUNT(*) FROM predictions WHERE run_id=:r"), {"r": run_id}).scalar_one()
            n_imp = c.execute(text("SELECT COUNT(*) FROM feature_importance WHERE run_id=:r"), {"r": run_id}).scalar_one()
            live = c.execute(text("SELECT actual_return, prob_up FROM predictions WHERE run_id=:r "
                                  "ORDER BY date DESC LIMIT 1"), {"r": run_id}).one()
        assert n_metrics > 15 and n_imp == len(FEATURE_COLUMNS)
        assert n_pred == len(res["oof"]) + 1               # out-of-sample days + the live forecast
        assert live.actual_return is None and 0 <= live.prob_up <= 1
    finally:
        with engine.begin() as c:
            for t in ("model_results", "predictions", "feature_importance"):
                c.execute(text(f"DELETE FROM {t} WHERE run_id=:r"), {"r": run_id})
