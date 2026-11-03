"""Evaluation metrics, kept in one place so every model is scored identically.

Classification metrics (XGBoost: "will tomorrow be up?") and regression metrics
(LSTM: "what will tomorrow's return be?") answer different questions and must
not be compared with each other. Directional accuracy is the one bridge: it asks
only "did the model get the SIGN right?", so both model types can be scored on it.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)


def directional_accuracy(actual_return, predicted) -> float:
    """Share of days where (predicted > 0) matches (actual return > 0).

    Pass predicted returns (regression) or 0/1 up-flags (classification).
    NOTE: for a classifier trained on "return > 0" this equals plain accuracy.
    """
    actual_up = np.asarray(actual_return) > 0
    pred_up = np.asarray(predicted) > 0
    return float((actual_up == pred_up).mean())


def classification_metrics(y_true, y_pred, prob_up=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "share_up_days": float(np.mean(y_true)),   # context: accuracy of "always up"
    }
    if prob_up is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, prob_up)
    return {k: float(v) for k, v in out.items()}


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
