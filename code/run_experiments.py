"""Experiments: eta x n_1 x lambda grid over 10 init seeds, validation-only selection,
then ONE test evaluation of the selected MLP and both baselines.

Run:  python code/run_experiments.py
Outputs (results/): grid_runs.csv, grid_summary.csv, loss_curves.csv, selection.json,
split_indices.csv, predictions.csv, test_metrics.csv, val_metrics.csv, run_info.json,
baseline_logreg_validation.csv.

Selection rule (validation only): minimize MEAN over seeds 0-9 of validation BCE (data term,
no regularizer, so lambdas are comparable). The final model uses init seed 0 (fixed in advance).
Predictions on the test set are computed in exactly one place: final_test_evaluation().
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn import metrics as skm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import baselines as bl  # noqa: E402
from exp_common import get_data, log_loss, metrics  # noqa: E402
from mlp import MLP  # noqa: E402

RES = cfg.RESULTS_DIR
FINAL_SEED = 0


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def keep_epoch(k: int, n: int) -> bool:
    """Subsample curves: every epoch up to 200, then every 10th, plus the last."""
    return k <= 200 or k % 10 == 0 or k == n - 1


def train_one(data, eta: float, n1: int, lam: float, seed: int):
    model = MLP([cfg.N_FEATURES, n1, 1], cfg.HIDDEN_ACTIVATION, lam=lam,
                init_scale=cfg.INIT_SCALE, seed=seed)
    t0 = time.perf_counter()
    h = model.fit(data.X_train, data.y_train, eta=eta, max_epochs=cfg.MAX_EPOCHS, tol=cfg.GRAD_TOL)
    return model, h, time.perf_counter() - t0


def run_grid(data):
    runs, curves = [], []
    for eta in cfg.LEARNING_RATES:
        for n1 in cfg.HIDDEN_WIDTHS:
            for lam in cfg.LAMBDAS:
                for seed in cfg.INIT_SEEDS:
                    m, h, dt = train_one(data, eta, n1, lam, seed)
                    pv, pt = m.predict_proba(data.X_val), m.predict_proba(data.X_train)
                    runs.append({
                        "eta": eta, "n1": n1, "lambda": lam, "seed": seed, "P": m.n_params,
                        "n_iter": h.n_iter, "converged": int(h.converged),
                        "hit_max_epochs": int(not h.converged and h.n_iter == cfg.MAX_EPOCHS),
                        "final_J_train": h.loss[-1], "final_grad_norm": h.grad_norm[-1],
                        "train_bce": m.loss(data.X_train, data.y_train, include_reg=False),
                        "train_acc": float(np.mean((pt >= .5) == data.y_train)),
                        "val_bce": m.loss(data.X_val, data.y_val, include_reg=False),
                        "val_acc": float(np.mean((pv >= .5) == data.y_val)),
                        "finite": int(np.isfinite(h.loss[-1])), "runtime_s": dt})
                    if seed == FINAL_SEED:
                        n = len(h.loss)
                        for k in range(n):
                            if keep_epoch(k, n):
                                curves.append({"eta": eta, "n1": n1, "lambda": lam, "seed": seed,
                                               "epoch": k, "J": h.loss[k], "grad_norm": h.grad_norm[k]})
        print(f"  eta={eta} done", flush=True)
    return runs, curves


def summarize(runs):
    out = []
    keys = sorted({(r["eta"], r["n1"], r["lambda"]) for r in runs})
    for eta, n1, lam in keys:
        rr = [r for r in runs if (r["eta"], r["n1"], r["lambda"]) == (eta, n1, lam)]
        vb = np.array([r["val_bce"] for r in rr]); va = np.array([r["val_acc"] for r in rr])
        out.append({"eta": eta, "n1": n1, "lambda": lam, "P": rr[0]["P"], "n_seeds": len(rr),
                    "val_bce_mean": vb.mean(), "val_bce_std": vb.std(ddof=1), "val_bce_min": vb.min(),
                    "val_bce_max": vb.max(), "val_acc_mean": va.mean(), "val_acc_min": va.min(),
                    "val_acc_max": va.max(),
                    "train_bce_mean": np.mean([r["train_bce"] for r in rr]),
                    "n_converged": sum(r["converged"] for r in rr),
                    "n_hit_max": sum(r["hit_max_epochs"] for r in rr),
                    "epochs_mean": np.mean([r["n_iter"] for r in rr]),
                    "final_grad_norm_mean": np.mean([r["final_grad_norm"] for r in rr]),
                    "runtime_s_mean": np.mean([r["runtime_s"] for r in rr])})
    return out


def final_test_evaluation(data, sel, grid_runs, data_source):
    """The ONLY function that evaluates a model on X_test / y_test (main() reads only the test size and indices)."""
    eta, n1, lam = sel["eta"], sel["n1"], sel["lambda"]
    model, h, dt = train_one(data, eta, n1, lam, FINAL_SEED)
    ref = next(r for r in grid_runs if (r["eta"], r["n1"], r["lambda"], r["seed"]) == (eta, n1, lam, FINAL_SEED))
    assert h.loss[-1] == ref["final_J_train"], "retrain not identical to grid run"
    lr, lr_lam, lr_rows = bl.select_logreg(data)
    maj_label, prior = bl.majority_rule(data.y_train)

    P = {"val": {}, "test": {}}
    for split, X, y in (("val", data.X_val, data.y_val), ("test", data.X_test, data.y_test)):
        P[split] = {"y": y, "mlp": model.predict_proba(X), "lr": lr.predict_proba(X)[:, 1],
                    "maj_p": np.full(len(y), prior), "maj_label": np.full(len(y), maj_label)}

    rows_v, rows_t, pred_rows = [], [], []
    for split, store in (("val", rows_v), ("test", rows_t)):
        y = P[split]["y"]
        for name, p in (("MLP", P[split]["mlp"]), ("LogReg", P[split]["lr"]),
                        ("Majority", P[split]["maj_p"])):
            m = metrics(y, p) if name != "Majority" else metrics(y, np.full(len(y), float(maj_label)))
            if name == "Majority":  # label rule uses hard label; log loss from train prior
                m["log_loss"] = log_loss(y, P[split]["maj_p"])
            # independent cross-check of my metric code against sklearn (same predictions)
            yhat = (np.full(len(y), maj_label) if name == "Majority" else (p >= .5).astype(int))
            assert abs(m["accuracy"] - skm.accuracy_score(y, yhat)) < 1e-12
            assert abs(m["recall"] - skm.recall_score(y, yhat, zero_division=np.nan)) < 1e-12 or np.isnan(m["recall"])  # type: ignore[arg-type]
            assert abs(m["f1"] - skm.f1_score(y, yhat, zero_division=0)) < 1e-12 or np.isnan(m["f1"])  # type: ignore[arg-type]
            cm = skm.confusion_matrix(y, yhat, labels=[0, 1])
            assert (cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0]) == (m["TP"], m["FP"], m["TN"], m["FN"])
            if name != "Majority":
                assert abs(m["log_loss"] - skm.log_loss(y, p)) < 1e-9
            store.append({"model": name, **m})
    # MLP log loss from logits (stable) vs from saved probabilities
    ll_logit = model.loss(data.X_test, data.y_test, include_reg=False)
    idx = {"val": data.idx_val, "test": data.idx_test}
    for split in ("val", "test"):
        for i in range(len(P[split]["y"])):
            pred_rows.append({"split": split, "row_index": int(idx[split][i]),
                              "y_true": int(P[split]["y"][i]),
                              "mlp_p": repr(float(P[split]["mlp"][i])),
                              "mlp_label": int(P[split]["mlp"][i] >= .5),
                              "logreg_p": repr(float(P[split]["lr"][i])),
                              "logreg_label": int(P[split]["lr"][i] >= .5),
                              "majority_label": maj_label, "majority_p": repr(prior)})
    info = {"selected": sel, "final_seed": FINAL_SEED, "logreg_lambda": lr_lam,
            "logreg_validation_table": lr_rows, "majority_label": maj_label, "majority_prior": prior,
            "mlp_test_log_loss_from_logits": ll_logit, "mlp_n_iter": h.n_iter,
            "mlp_converged": h.converged, "mlp_final_grad_norm": h.grad_norm[-1],
            "mlp_runtime_s": dt, "mlp_P": model.n_params, "logreg_n_params": lr.coef_.size + 1}
    return rows_v, rows_t, pred_rows, info


def main() -> None:
    RES.mkdir(exist_ok=True)
    data, src = get_data()
    print(f"data source: {src}")
    print(f"train/val/test = {len(data.y_train)}/{len(data.y_val)}/{len(data.y_test)}")
    t0 = time.perf_counter()
    runs, curves = run_grid(data)
    grid_time = time.perf_counter() - t0
    summ = summarize(runs)
    write_csv(RES / "grid_runs.csv", runs)
    write_csv(RES / "grid_summary.csv", summ)
    write_csv(RES / "loss_curves.csv", curves)

    best = min(summ, key=lambda r: r["val_bce_mean"])
    ranked = sorted(summ, key=lambda r: r["val_bce_mean"])
    sel = {"eta": best["eta"], "n1": best["n1"], "lambda": best["lambda"],
           "criterion": "min mean over seeds 0-9 of validation BCE (no regularizer)",
           "val_bce_mean": best["val_bce_mean"], "val_acc_mean": best["val_acc_mean"],
           "runner_up": {k: ranked[1][k] for k in ("eta", "n1", "lambda", "val_bce_mean")}}
    json.dump(sel, open(RES / "selection.json", "w"), indent=2)
    print("selected on validation:", sel)

    rows_v, rows_t, pred_rows, info = final_test_evaluation(data, sel, runs, src)
    write_csv(RES / "val_metrics.csv", rows_v)
    write_csv(RES / "test_metrics.csv", rows_t)
    write_csv(RES / "predictions.csv", pred_rows)
    split_rows = [{"row_index": int(i), "split": s}
                  for s, ix in (("train", data.idx_train), ("val", data.idx_val), ("test", data.idx_test))
                  for i in ix]
    write_csv(RES / "split_indices.csv", sorted(split_rows, key=lambda r: r["row_index"]))
    write_csv(RES / "baseline_logreg_validation.csv", info["logreg_validation_table"])

    info.update({"data_source": src, "grid_total_runtime_s": grid_time, "n_runs": len(runs),
                 "python": platform.python_version(), "numpy": np.__version__,
                 "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                 "platform": platform.platform(), "processor": platform.processor(),
                 "max_epochs": cfg.MAX_EPOCHS, "grad_tol": cfg.GRAD_TOL, "init_scale": cfg.INIT_SCALE,
                 "split_seed": cfg.SPLIT_SEED, "init_seeds": list(cfg.INIT_SEEDS)})
    json.dump(info, open(RES / "run_info.json", "w"), indent=2, default=str)
    print("\nTEST (evaluated once):")
    for r in rows_t:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"grid runtime {grid_time:.1f}s for {len(runs)} runs")


if __name__ == "__main__":
    main()
