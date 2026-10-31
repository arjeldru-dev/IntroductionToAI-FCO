"""Shared helpers for the experiments: data access and metric computation.

get_data() uses the official data/wdbc.data via data_prep.prepare_data() when the file is
present (checksum verified). If it is absent it falls back to scikit-learn's bundled copy of
WDBC (same 569 x 30 values; label 1 = malignant) with the SAME split and scaler code, and
reports data_source = "sklearn_bundled_FALLBACK" so the caveat is recorded in every output.
"""
from __future__ import annotations

import sys
from pathlib import Path

from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import data_prep as dp  # noqa: E402

EPS = 1e-15  # clip for log loss computed from saved probabilities


def get_data() -> tuple[dp.Data, str]:
    """Return (Data, data_source)."""
    if cfg.DATA_FILE.exists():
        return dp.prepare_data(), "wdbc.data (sha256 verified)"
    from sklearn.datasets import load_breast_cancer

    b: Any = load_breast_cancer()
    X, y = b.data.astype(float), (1 - b.target).astype(int)  # sklearn: 0 = malignant
    tr, va, te = dp.stratified_split(y, cfg.SPLIT_SEED)
    sc = dp.Standardizer.fit(X[tr])
    d = dp.Data(sc.transform(X[tr]), y[tr], sc.transform(X[va]), y[va], sc.transform(X[te]),
                y[te], sc, tr, va, te, list(cfg.FEATURE_NAMES))
    return d, "sklearn_bundled_FALLBACK"


def log_loss(y: Any, p: Any) -> float:
    """Mean binary cross-entropy from probabilities, p clipped to [EPS, 1-EPS]."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def confusion(y: Any, yhat: Any) -> dict[str, int]:
    """TP, FP, TN, FN with positive = malignant (1)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    return {"TP": int(np.sum((y == 1) & (yhat == 1))), "FP": int(np.sum((y == 0) & (yhat == 1))),
            "TN": int(np.sum((y == 0) & (yhat == 0))), "FN": int(np.sum((y == 1) & (yhat == 0)))}


def metrics(y: Any, p: Any, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, precision, recall, F1, confusion counts, log loss. Undefined ratios -> nan."""
    yhat = (np.asarray(p) >= threshold).astype(int)
    c = confusion(y, yhat)
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    rec = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else float("nan")
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1,
            **c, "log_loss": log_loss(y, p)}
