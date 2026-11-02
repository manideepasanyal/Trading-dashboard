"""XGBoost classifier: will the NEXT trading day's return be positive?

Run:  python -m src.train_xgb                 (all tickers, 5 walk-forward folds)
      python -m src.train_xgb --symbols AAPL --folds 6

Validation is *expanding-window walk-forward*: train on everything before a
cut-off, predict the next block, then move the cut-off forward and repeat.
Every prediction is therefore made by a model that never saw that day or any
later day. Results are stored in PostgreSQL for the dashboard.
"""
import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import SYMBOLS
from .db import get_engine, init_schema, load_daily_prices, upsert_df
from .features import (FEATURE_COLUMNS, TARGET_RETURN, TARGET_UP, add_features,
                       split_labelled_and_latest)
from .metrics import classification_metrics

log = logging.getLogger("train_xgb")

# Deliberately small, regularised trees: daily returns are mostly noise, so a big
# model just memorises it. These are fixed, NOT tuned on the test folds.
XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=2,
)
MIN_ROWS = 300


def walk_forward_splits(n: int, n_folds: int = 5, initial_frac: float = 0.5):
    """Yield (train_idx, test_idx) with an expanding train window.

    The first `initial_frac` of the data is only ever training data. The rest is
    cut into `n_folds` consecutive test blocks. Train indices always end right
    before the test block starts.

    Why no gap/embargo? The label of row t is the return from close t to close
    t+1. For the last training row (start-1) that return is realised at the close
    of day `start`, exactly when the first test prediction is made. Nothing from
    the future is used. (With multi-day horizons you WOULD need a gap.)
    """
    first_test = int(n * initial_frac)
    fold_size = (n - first_test) // n_folds
    for i in range(n_folds):
        start = first_test + i * fold_size
        end = n if i == n_folds - 1 else start + fold_size
        yield np.arange(0, start), np.arange(start, end)


def run_walk_forward(labelled: pd.DataFrame, n_folds: int = 5,
                     feature_columns=None, params=None) -> dict:
    """Walk-forward evaluation + final model. Pure function: no database access."""
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    params = params or XGB_PARAMS
    if len(labelled) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} labelled rows, got {len(labelled)}")

    X = labelled[feature_columns]
    y = labelled[TARGET_UP].astype(int)

    folds = []
    for i, (tr, te) in enumerate(walk_forward_splits(len(labelled), n_folds)):
        model = XGBClassifier(**params).fit(X.iloc[tr], y.iloc[tr])
        folds.append(pd.DataFrame({
            "date": labelled["date"].iloc[te].values,
            "actual_return": labelled[TARGET_RETURN].iloc[te].values,
            "actual_up": y.iloc[te].values,
            "prob_up": model.predict_proba(X.iloc[te])[:, 1],
            # Baselines see only information available at prediction time:
            "pred_majority": int(y.iloc[tr].mean() >= 0.5),          # most common direction in training
            "pred_persistence": (labelled["ret_1d"].iloc[te] > 0).astype(int).values,  # tomorrow = today
            "fold": i,
        }))
    oof = pd.concat(folds, ignore_index=True)
    oof["pred_xgb"] = (oof["prob_up"] >= 0.5).astype(int)

    metrics = {
        "xgboost": classification_metrics(oof["actual_up"], oof["pred_xgb"], oof["prob_up"]),
        "baseline_majority": classification_metrics(oof["actual_up"], oof["pred_majority"]),
        "baseline_persistence": classification_metrics(oof["actual_up"], oof["pred_persistence"]),
    }
    fold_acc = oof.groupby("fold").apply(
        lambda g: (g["pred_xgb"] == g["actual_up"]).mean(), include_groups=False)
    metrics["xgboost"]["fold_accuracy_std"] = float(fold_acc.std(ddof=0))
    metrics["xgboost"]["n_test_days"] = float(len(oof))

    final = XGBClassifier(**params).fit(X, y)   # trained on all labelled history, for the live forecast
    importance = pd.Series(final.feature_importances_, index=feature_columns).sort_values(ascending=False)
    return {"oof": oof, "metrics": metrics, "importance": importance, "final_model": final,
            "feature_columns": feature_columns}


def predict_latest(result: dict, latest: pd.DataFrame) -> float | None:
    """P(up) for the most recent day (features known, outcome not yet)."""
    if latest.empty:
        return None
    return float(result["final_model"].predict_proba(latest[result["feature_columns"]])[:, 1][0])


def save_results(engine, run_id: str, symbol: str, result: dict,
                 latest: pd.DataFrame | None = None) -> None:
    metric_rows = [
        {"run_id": run_id, "symbol": symbol, "model": model,
         "task": "classification", "split": "walk_forward", "metric": m, "value": v}
        for model, ms in result["metrics"].items() for m, v in ms.items()
    ]
    pd.DataFrame(metric_rows).to_sql("model_results", engine, if_exists="append", index=False)

    oof = result["oof"]
    pred = pd.DataFrame({
        "run_id": run_id, "symbol": symbol, "model": "xgboost",
        "date": pd.to_datetime(oof["date"]).dt.date,
        "actual_return": oof["actual_return"], "predicted_return": np.nan,
        "prob_up": oof["prob_up"],
    })
    if latest is not None and not latest.empty:
        prob = predict_latest(result, latest)
        pred = pd.concat([pred, pd.DataFrame([{
            "run_id": run_id, "symbol": symbol, "model": "xgboost",
            "date": latest["date"].iloc[0].date(), "actual_return": np.nan,
            "predicted_return": np.nan, "prob_up": prob}])], ignore_index=True)
    upsert_df(engine, "predictions", pred, key_cols=["run_id", "symbol", "model", "date"])

    imp = pd.DataFrame({"run_id": run_id, "symbol": symbol, "model": "xgboost",
                        "feature": result["importance"].index,
                        "importance": result["importance"].values})
    upsert_df(engine, "feature_importance", imp, key_cols=["run_id", "symbol", "model", "feature"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = get_engine()
    init_schema(engine)
    run_id = datetime.now(timezone.utc).strftime("xgb_%Y%m%d%H%M%S")
    failures = 0
    for symbol in args.symbols:
        try:
            prices = load_daily_prices(engine, symbol)
            labelled, latest = split_labelled_and_latest(add_features(prices))
            result = run_walk_forward(labelled, args.folds)
            save_results(engine, run_id, symbol, result, latest)
            m = result["metrics"]
            print(f"\n{symbol}  ({int(m['xgboost']['n_test_days'])} out-of-sample days, {args.folds} folds)")
            for model in ("xgboost", "baseline_majority", "baseline_persistence"):
                x = m[model]
                print(f"  {model:22s} acc={x['accuracy']:.3f} prec={x['precision']:.3f} "
                      f"rec={x['recall']:.3f} f1={x['f1']:.3f}")
            print("  top features:", ", ".join(result["importance"].head(3).index))
        except Exception as exc:
            failures += 1
            log.error("%s failed: %s", symbol, exc)
    return 1 if failures == len(args.symbols) else 0


if __name__ == "__main__":
    sys.exit(main())
