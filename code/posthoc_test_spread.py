"""POST-HOC analysis (run after the main selection was fixed): test-set spread of the SELECTED
MLP configuration over init seeds 0-9.

This is a SECOND look at the test set (the first is final_test_evaluation() in
run_experiments.py) and is reported as post-hoc. It must not feed any decision: the
configuration comes from results/selection.json (validation only) and is NOT re-selected here.
Seed 0 must reproduce the MLP row of results/test_metrics.csv exactly (checked below).

Run:  python code/posthoc_test_spread.py     (after run_experiments.py)
Output: results/posthoc_test_spread.csv (one row per seed, then mean/std/min/max rows)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
from exp_common import get_data, metrics  # noqa: E402
from run_experiments import FINAL_SEED, train_one  # noqa: E402

RES = cfg.RESULTS_DIR
KEYS = ["accuracy", "precision", "recall", "f1", "TP", "FP", "TN", "FN", "log_loss"]


def main() -> None:
    data, src = get_data()
    if not src.startswith("wdbc.data"):
        raise SystemExit(f"refusing to run: data source is {src!r}, expected the verified wdbc.data")
    sel = json.load(open(RES / "selection.json"))
    eta, n1, lam = sel["eta"], sel["n1"], sel["lambda"]
    print(f"POST-HOC test spread, selected config eta={eta} n1={n1} lambda={lam}, data: {src}")
    rows = []
    for seed in cfg.INIT_SEEDS:
        model, h, _ = train_one(data, eta, n1, lam, seed)
        m = metrics(data.y_test, model.predict_proba(data.X_test))
        rows.append({"analysis": "post-hoc", "seed": seed, "n_iter": h.n_iter, "converged": int(h.converged),
                     "final_grad_norm": h.grad_norm[-1], **{k: m[k] for k in KEYS}})
    # reproduce the single official evaluation (seed FINAL_SEED) before reporting anything else
    with open(RES / "test_metrics.csv", newline="") as f:
        official = next(r for r in csv.DictReader(f) if r["model"] == "MLP")
    mine = next(r for r in rows if r["seed"] == FINAL_SEED)
    for k in KEYS:
        assert float(official[k]) == float(mine[k]) or (np.isnan(float(official[k])) and np.isnan(mine[k])), k
    print(f"seed {FINAL_SEED} reproduces results/test_metrics.csv exactly")
    for stat, fn in (("mean", np.mean), ("std_ddof1", lambda v: np.std(v, ddof=1)), ("min", np.min), ("max", np.max)):
        rows.append({"analysis": "post-hoc", "seed": stat, "n_iter": "", "converged": "", "final_grad_norm": "",
                     **{k: float(fn([r[k] for r in rows if isinstance(r["seed"], int)])) for k in KEYS}})
    with open(RES / "posthoc_test_spread.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
               if k in ("seed", "converged", "accuracy", "recall", "TP", "FP", "TN", "FN", "log_loss")})


if __name__ == "__main__":
    main()
