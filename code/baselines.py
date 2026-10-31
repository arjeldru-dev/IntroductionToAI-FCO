"""Baselines on the IDENTICAL split and scaling as the MLP.

1. Majority-class rule: always predict the most frequent TRAIN class. Its probability output
   is the constant TRAIN malignant fraction (used only for log loss).
2. Logistic regression (scikit-learn, L2). Penalty convention: sklearn minimizes
       C * sum_r BCE^(r) + 0.5 ||w||_2^2      (intercept NOT penalized with lbfgs)
   Dividing by C*N_train gives  mean BCE + (1/(2 C N)) ||w||^2, which equals our
   J = mean BCE + (lambda/2)||w||^2 when  C = 1 / (N_train * lambda).  For lambda = 0 we use
   C = 1e10 (effectively unpenalized). The mapping is verified numerically (check_convention).
   lambda is chosen on VALIDATION log loss over the same grid as the MLP.

Run:  python code/baselines.py      (validation only; the test set is evaluated in run_experiments.py)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
from exp_common import get_data, log_loss, metrics  # noqa: E402
from mlp import MLP  # noqa: E402

C_UNPENALIZED = 1e10


def c_from_lambda(lam: float, n_train: int) -> float:
    return C_UNPENALIZED if lam == 0 else 1.0 / (n_train * lam)


def fit_logreg(X: np.ndarray, y: np.ndarray, lam: float) -> LogisticRegression:
    m = LogisticRegression(C=c_from_lambda(lam, len(y)), tol=1e-12, max_iter=100000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return m


def majority_rule(y_train: np.ndarray) -> tuple[int, float]:
    """(majority label, train malignant fraction)."""
    frac = float(np.mean(y_train))
    return int(frac >= 0.5), frac


def check_convention(X: np.ndarray, y: np.ndarray, lam: float, model: LogisticRegression) -> float:
    """||grad J||_2 of OUR objective (30-1 'MLP' = logistic regression) at sklearn's solution."""
    net = MLP([X.shape[1], 1], lam=lam, seed=0)
    net.W[0][...] = model.coef_
    net.b[0][...] = np.asarray(model.intercept_).reshape(1, 1)
    _, g = net.loss_and_grads(X.T, y)
    return float(np.linalg.norm(net.flatten_grads(g)))


def select_logreg(data) -> tuple[LogisticRegression, float, list[dict]]:
    """Fit over cfg.LAMBDAS, pick lowest validation log loss. Returns (model, lam, table)."""
    rows, best = [], None
    for lam in cfg.LAMBDAS:
        m = fit_logreg(data.X_train, data.y_train, lam)
        pv = m.predict_proba(data.X_val)[:, 1]
        vl = log_loss(data.y_val, pv)
        gn = check_convention(data.X_train, data.y_train, lam, m)
        rows.append({"lambda": lam, "C": c_from_lambda(lam, len(data.y_train)), "val_log_loss": vl,
                     "val_accuracy": metrics(data.y_val, pv)["accuracy"],
                     "train_grad_norm_of_our_J": gn, "n_iter_sklearn": int(m.n_iter_[0])})
        if best is None or vl < best[0]:
            best = (vl, lam, m)
    assert best is not None
    return best[2], best[1], rows


def main() -> None:
    import csv
    data, src = get_data()
    print(f"data source: {src}")
    m, lam, rows = select_logreg(data)
    lab, frac = majority_rule(data.y_train)
    print(f"majority rule: predict {lab} (benign=0), train malignant fraction {frac:.4f}")
    for r in rows:
        print({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"selected lambda (validation log loss): {lam}")
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    with open(cfg.RESULTS_DIR / "baseline_logreg_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
