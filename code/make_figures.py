"""Build all figures from results/*.csv (never from live training). 300 dpi PNGs in results/.

Run:  python code/make_figures.py      (after run_experiments.py)
Also writes results/roc_pr_summary.csv (AUC, average precision on the TEST predictions;
no model is re-evaluated: it only reads results/predictions.csv).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

RES = cfg.RESULTS_DIR
DPI = 300
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def read(name: str) -> list[dict]:
    with open(RES / name, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(RES / name, dpi=DPI)
    plt.close(fig)
    print("wrote", name)


def curves(cv, n1, lam, key):
    out = {}
    for eta in cfg.LEARNING_RATES:
        rows = [r for r in cv if float(r["eta"]) == eta and int(r["n1"]) == n1 and float(r["lambda"]) == lam]
        out[eta] = (np.array([int(r["epoch"]) for r in rows]), np.array([float(r[key]) for r in rows]))
    return out


def fig_eta_curves(cv, n1=8, lam=0.0):
    ref = fr"reference config $n_1$ = {n1}, $\lambda$ = {lam:g} (not the selected config), init seed 0"
    for key, ylabel, title, legend_loc, name in (
            ("J", "training objective J", "Training objective", "upper right", "fig_loss_1.png"),
            ("grad_norm", r"$\|\nabla J\|_2$", "Gradient norm", "empty band", "fig_gradnorm_2.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for eta, (x, y) in curves(cv, n1, lam, key).items():
            ax.plot(x, y, label=fr"$\eta$ = {eta:g}")
        if key == "grad_norm":
            ax.axhline(cfg.GRAD_TOL, color="k", ls="--", lw=1, label=f"tolerance {cfg.GRAD_TOL:g}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch (full-batch iteration)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{ref}", fontsize=9)
        if legend_loc == "empty band":  # between the curves and the tolerance line; covers neither
            ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.30))
        else:
            ax.legend(loc=legend_loc)
        save(fig, name)


def fig_val_heatmap(summ, sel):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    vmin = min(float(r["val_bce_mean"]) for r in summ)
    vmax = max(float(r["val_bce_mean"]) for r in summ)  # full range: no saturated cells
    im = None
    for ax, eta in zip(axes, cfg.LEARNING_RATES):
        M = np.zeros((len(cfg.HIDDEN_WIDTHS), len(cfg.LAMBDAS)))
        for i, n1 in enumerate(cfg.HIDDEN_WIDTHS):
            for j, lam in enumerate(cfg.LAMBDAS):
                r = next(r for r in summ if float(r["eta"]) == eta and int(r["n1"]) == n1 and float(r["lambda"]) == lam)
                M[i, j] = float(r["val_bce_mean"])
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis_r", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                chosen = (eta, cfg.HIDDEN_WIDTHS[i], cfg.LAMBDAS[j]) == (sel["eta"], sel["n1"], sel["lambda"])
                ax.text(j, i, f"{M[i, j]:.3f}" + ("\n(selected)" if chosen else ""), ha="center",
                        va="center", color="w" if M[i, j] > (vmin + vmax) / 2 else "k", fontsize=8,
                        fontweight="bold" if chosen else "normal")
        ax.set_xticks(range(3), [f"{l:g}" for l in cfg.LAMBDAS])
        ax.set_yticks(range(3), [str(n) for n in cfg.HIDDEN_WIDTHS])
        ax.set_xlabel(r"$\lambda$")
        ax.set_title(fr"$\eta$ = {eta:g}")
        ax.grid(False)
    axes[0].set_ylabel(r"hidden width $n_1$")
    assert im is not None
    fig.colorbar(im, ax=axes, shrink=0.85, label="mean validation BCE (10 seeds)")
    fig.savefig(RES / "fig_val_1.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_val_1.png")


def fig_seed_spread(runs, sel):
    eta = sel["eta"]
    fig, ax = plt.subplots(figsize=(8, 4))
    pos, data, labels = [], [], []
    k = 0
    for n1 in cfg.HIDDEN_WIDTHS:
        for lam in cfg.LAMBDAS:
            v = [float(r["val_bce"]) for r in runs if float(r["eta"]) == eta and int(r["n1"]) == n1
                 and float(r["lambda"]) == lam]
            data.append(v); pos.append(k); labels.append(f"{n1}\n{lam:g}"); k += 1
        k += 0.6
    ax.boxplot(data, positions=pos, widths=0.6)
    for p, v in zip(pos, data):
        ax.scatter(np.full(len(v), p), v, s=12, color="C1", zorder=3)
    ax.set_xticks(pos, labels)
    ax.set_xlabel(r"$n_1$ (top) and $\lambda$ (bottom)")
    ax.set_ylabel("validation BCE")
    ax.set_title(fr"Spread over init seeds 0-9, $\eta$ = {eta:g}")
    save(fig, "fig_seeds_1.png")


def fig_confusion(preds):
    te = [r for r in preds if r["split"] == "test"]
    y = np.array([int(r["y_true"]) for r in te])
    cols = (("MLP", "mlp_label"), ("Logistic regression", "logreg_label"), ("Majority class", "majority_label"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (nm, c) in zip(axes, cols):
        yh = np.array([int(r[c]) for r in te])
        cm = skm.confusion_matrix(y, yh, labels=[1, 0])  # rows: true M, B; cols: pred M, B
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                        color="w" if cm[i, j] > cm.max() / 2 else "k")
        ax.set_xticks([0, 1], ["malignant", "benign"]); ax.set_yticks([0, 1], ["malignant", "benign"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(nm); ax.grid(False)
    fig.suptitle("Test-set confusion matrices (positive = malignant)")
    save(fig, "fig_confusion_1.png")


def fig_roc_pr(preds):
    te = [r for r in preds if r["split"] == "test"]
    y = np.array([int(r["y_true"]) for r in te])
    prevalence = float(y.mean())
    fig1, a1 = plt.subplots(figsize=(4.8, 4.4)); fig2, a2 = plt.subplots(figsize=(4.8, 4.4))
    styles: dict[str, dict[str, Any]] = {
        "MLP": dict(color="C0", ls="-", lw=3.5, alpha=0.9),
        "Logistic regression": dict(color="C1", ls=(0, (4, 3)), lw=1.8),
    }
    rows, roc_pts, pr_pts = [], {}, {}
    for nm, c in (("MLP", "mlp_p"), ("Logistic regression", "logreg_p")):
        p = np.array([float(r[c]) for r in te])
        fpr, tpr, _ = skm.roc_curve(y, p); pr, rc, _ = skm.precision_recall_curve(y, p)
        auc, ap = skm.roc_auc_score(y, p), skm.average_precision_score(y, p)
        a1.plot(fpr, tpr, label=f"{nm} (AUC {auc:.4f})", **styles[nm])
        a2.plot(rc, pr, label=f"{nm} (AP {ap:.4f})", **styles[nm])
        roc_pts[nm], pr_pts[nm] = (fpr, tpr), (rc, pr)
        rows.append({"model": nm, "roc_auc": auc, "average_precision": ap})
    same_roc = all(np.array_equal(u, v) for u, v in zip(roc_pts["MLP"], roc_pts["Logistic regression"]))
    same_pr = all(np.array_equal(u, v) for u, v in zip(pr_pts["MLP"], pr_pts["Logistic regression"]))
    a1.plot([0, 1], [0, 1], color="k", ls=":", lw=1, label="chance")
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate")
    a1.set_title("ROC, test set" + ("\n(the two curves coincide)" if same_roc else ""), fontsize=10)
    a1.legend(loc="lower right")
    a2.axhline(prevalence, color="k", ls=":", lw=1, label=f"prevalence baseline ({prevalence:.3f})")
    a2.set_xlabel("recall"); a2.set_ylabel("precision"); a2.set_ylim(0, 1.05)
    a2.set_title("Precision-recall, test set" + ("\n(the two curves coincide)" if same_pr else ""), fontsize=10)
    a2.legend(loc="lower left")
    save(fig1, "fig_roc_1.png"); save(fig2, "fig_pr_2.png")
    with open(RES / "roc_pr_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(rows, "| curves identical: ROC", same_roc, "PR", same_pr, "| prevalence", round(prevalence, 4))


def main() -> None:
    sel = json.load(open(RES / "selection.json"))
    cv, summ, runs, preds = read("loss_curves.csv"), read("grid_summary.csv"), read("grid_runs.csv"), read("predictions.csv")
    fig_eta_curves(cv)
    fig_val_heatmap(summ, sel)
    fig_seed_spread(runs, sel)
    fig_confusion(preds)
    fig_roc_pr(preds)


if __name__ == "__main__":
    main()
