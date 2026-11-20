"""Threshold selection on validation data and test metrics (Sec. 4.4)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def select_threshold(y_true: Sequence[int], y_prob: Sequence[float], metric: str = "f1") -> float:
    """Pick tau on validation data only. Never call this with test data."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.min() == y_true.max():
        return 0.5  # single-class validation set: no information to tune on
    if metric == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))
    if metric == "f1":
        precision, recall, thr = precision_recall_curve(y_true, y_prob)
        p, r = precision[:-1], recall[:-1]  # last point has no threshold
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        return float(thr[np.argmax(f1)])
    raise ValueError(f"Unknown threshold metric {metric!r}")


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float], tau: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out: dict[str, float] = {"threshold": float(tau), "n": int(len(y_true)),
                             "positive_rate": float(y_true.mean()) if len(y_true) else 0.0}
    out.update(_confusion(y_true, (y_prob >= tau).astype(int)))
    both_classes = len(np.unique(y_true)) == 2
    out["auroc"] = float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan")
    out["auprc"] = float(average_precision_score(y_true, y_prob)) if both_classes else float("nan")
    return out


def metrics_by_length(
    y_true: Sequence[int], y_prob: Sequence[float], target_lengths: Sequence[int], tau: float
) -> dict[int, dict[str, float]]:
    """Per-window-length breakdown, incl. the malicious proportion at each length (Sec. 4.4)."""
    y_true, y_prob, lengths = map(np.asarray, (y_true, y_prob, target_lengths))
    return {
        int(L): compute_metrics(y_true[lengths == L], y_prob[lengths == L], tau)
        for L in np.unique(lengths)
    }
