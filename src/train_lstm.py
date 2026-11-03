"""PyTorch LSTM: forecast the NEXT trading day's return from the previous 30 days.

Run:  python -m src.train_lstm                 (all tickers)
      python -m src.train_lstm --symbols AAPL --epochs 60

Setup, in the order that avoids look-ahead bias:
1. Build one sample per day t: input = features of days t-29..t, target = return t -> t+1.
2. Split the samples CHRONOLOGICALLY: first 70% train, next 15% validation, last 15% test.
3. Fit the feature scaler (and target scale) on TRAIN rows only, then apply to the rest.
4. Train on train; use validation loss for early stopping; touch the test set once, at the end.
Results are stored in PostgreSQL for the dashboard.
"""
import argparse
import copy
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import SYMBOLS
from .db import get_engine, init_schema, load_daily_prices, upsert_df
from .features import FEATURE_COLUMNS, TARGET_RETURN, add_features, split_labelled_and_latest
from .metrics import regression_metrics

log = logging.getLogger("train_lstm")

SEQ_LEN = 30
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
MIN_SAMPLES = 300
SEED = 42


class LSTMRegressor(nn.Module):
    """Small on purpose: one LSTM layer + a linear head. Daily returns are noisy,
    so extra capacity mostly memorises noise."""

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)   # use the last time step


def make_windows(values: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """values: (n_rows, n_feat) -> (n_rows - seq_len + 1, seq_len, n_feat).
    Window k covers rows k..k+seq_len-1, so it ENDS at row k+seq_len-1."""
    idx = np.arange(seq_len)[None, :] + np.arange(len(values) - seq_len + 1)[:, None]
    return values[idx]


def chronological_split(n_samples: int):
    n_train = int(n_samples * TRAIN_FRAC)
    n_val = int(n_samples * VAL_FRAC)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_samples)


def _predict(model, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


def train_lstm(labelled: pd.DataFrame, feature_columns=None, epochs: int = 40,
               patience: int = 8, seq_len: int = SEQ_LEN, hidden: int = 32,
               lr: float = 1e-3, batch_size: int = 64) -> dict:
    """Train + evaluate. Pure function: no database access."""
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    if len(labelled) - seq_len + 1 < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} samples, got {len(labelled) - seq_len + 1}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    feats = labelled[feature_columns].to_numpy(dtype=np.float64)
    target = labelled[TARGET_RETURN].to_numpy(dtype=np.float64)
    n_samples = len(labelled) - seq_len + 1
    tr, va, te = chronological_split(n_samples)

    # Sample k ends at row k + seq_len - 1. Rows the TRAIN samples can see: 0 .. tr.stop+seq_len-2.
    train_last_row = tr.stop + seq_len - 2
    mean = feats[: train_last_row + 1].mean(axis=0)
    std = feats[: train_last_row + 1].std(axis=0)
    std[std == 0] = 1.0
    X_all = make_windows((feats - mean) / std, seq_len)
    y_all = target[seq_len - 1:]                         # target of the window's last row
    dates = labelled["date"].to_numpy()[seq_len - 1:]

    y_scale = y_all[tr].std() or 1.0                     # train-only; keeps the loss near 1
    Xt = torch.tensor(X_all, dtype=torch.float32)
    yt = torch.tensor(y_all / y_scale, dtype=torch.float32)

    model = LSTMRegressor(len(feature_columns), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(np.arange(tr.start, tr.stop))   # shuffle within TRAIN only
        for b in range(0, len(order), batch_size):
            ids = order[b:b + batch_size]
            opt.zero_grad()
            loss_fn(model(Xt[ids]), yt[ids]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_loss = float(loss_fn(torch.tensor(_predict(model, Xt[va])), yt[va]))
        history.append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)                    # keep the best validation epoch

    pred = _predict(model, Xt) * y_scale                 # back to real return units
    train_mean = float(y_all[tr].mean())
    metrics = {}
    for split_name, sl in (("val", va), ("test", te)):
        actual = y_all[sl]
        majority_up = float(y_all[tr].mean() > 0)        # most common training direction proxy
        metrics[split_name] = {
            "lstm": regression_metrics(actual, pred[sl]),
            "baseline_zero": regression_metrics(actual, np.zeros_like(actual)),
            "baseline_train_mean": regression_metrics(actual, np.full_like(actual, train_mean)),
        }
        # Direction of a constant baseline = share of days it gets right by always saying "up"/"down".
        metrics[split_name]["baseline_train_mean"]["directional_accuracy"] = float(
            ((actual > 0) == (train_mean > 0)).mean())
        metrics[split_name]["baseline_zero"]["directional_accuracy"] = float((actual <= 0).mean())
    metrics["test"]["lstm"]["n_test_days"] = float(len(y_all[te]))
    metrics["test"]["lstm"]["epochs_run"] = float(len(history))

    return {"model": model, "metrics": metrics, "mean": mean, "std": std, "y_scale": y_scale,
            "feature_columns": feature_columns, "seq_len": seq_len,
            "test_dates": dates[te], "test_actual": y_all[te], "test_pred": pred[te],
            "val_history": history}


def predict_latest(result: dict, labelled: pd.DataFrame, latest: pd.DataFrame) -> float | None:
    """Forecast for the most recent day: window = last 30 days of features (the latest row included)."""
    if latest.empty:
        return None
    cols, seq_len = result["feature_columns"], result["seq_len"]
    rows = pd.concat([labelled[cols].tail(seq_len - 1), latest[cols]]).to_numpy(dtype=np.float64)
    if len(rows) < seq_len:
        return None
    window = ((rows - result["mean"]) / result["std"])[None, :, :]
    out = _predict(result["model"], torch.tensor(window, dtype=torch.float32))
    return float(out[0] * result["y_scale"])


def save_results(engine, run_id: str, symbol: str, result: dict, labelled=None, latest=None) -> None:
    rows = [{"run_id": run_id, "symbol": symbol, "model": model, "task": "regression",
             "split": split, "metric": m, "value": v}
            for split, per_model in result["metrics"].items()
            for model, ms in per_model.items() for m, v in ms.items()]
    pd.DataFrame(rows).to_sql("model_results", engine, if_exists="append", index=False)

    pred = pd.DataFrame({
        "run_id": run_id, "symbol": symbol, "model": "lstm",
        "date": pd.to_datetime(result["test_dates"]).date,
        "actual_return": result["test_actual"], "predicted_return": result["test_pred"],
        "prob_up": np.nan})
    if labelled is not None and latest is not None and not latest.empty:
        live = predict_latest(result, labelled, latest)
        if live is not None:
            pred = pd.concat([pred, pd.DataFrame([{
                "run_id": run_id, "symbol": symbol, "model": "lstm",
                "date": latest["date"].iloc[0].date(), "actual_return": np.nan,
                "predicted_return": live, "prob_up": np.nan}])], ignore_index=True)
    upsert_df(engine, "predictions", pred, key_cols=["run_id", "symbol", "model", "date"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = get_engine()
    init_schema(engine)
    run_id = datetime.now(timezone.utc).strftime("lstm_%Y%m%d%H%M%S")
    failures = 0
    for symbol in args.symbols:
        try:
            labelled, latest = split_labelled_and_latest(add_features(load_daily_prices(engine, symbol)))
            result = train_lstm(labelled, epochs=args.epochs)
            save_results(engine, run_id, symbol, result, labelled, latest)
            t = result["metrics"]["test"]
            print(f"\n{symbol}  (test: {int(t['lstm']['n_test_days'])} days, "
                  f"{int(t['lstm']['epochs_run'])} epochs)")
            for model in ("lstm", "baseline_zero", "baseline_train_mean"):
                x = t[model]
                print(f"  {model:20s} MAE={x['mae']:.5f} RMSE={x['rmse']:.5f} "
                      f"directional_acc={x['directional_accuracy']:.3f}")
        except Exception as exc:
            failures += 1
            log.error("%s failed: %s", symbol, exc)
    return 1 if failures == len(args.symbols) else 0


if __name__ == "__main__":
    sys.exit(main())
