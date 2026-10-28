"""Centered finite-difference gradient check for the from-scratch MLP.

(J(theta + h e_i) - J(theta - h e_i)) / (2h) is compared with the backprop
gradient at ALL P parameters (h = 1e-5).  Required: max absolute
gap < 1e-6.

Run:  python code/gradient_check.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp import MLP  # noqa: E402

H_FD = 1e-5  # finite-difference step h; do not confuse with hidden width n_1
GAP_TOL = 1e-6


def finite_difference_gradient(model: MLP, X: np.ndarray, y: np.ndarray, h: float = H_FD) -> np.ndarray:
    """Centered finite-difference gradient of J at every parameter.

    ``X`` is (N, d), ``y`` is (N,).  The model's parameters are restored afterwards.
    """
    theta0 = model.get_params()
    fd = np.empty_like(theta0)
    try:
        for i in range(theta0.size):
            th = theta0.copy()
            th[i] = theta0[i] + h
            model.set_params(th)
            j_plus = model.loss(X, y)
            th[i] = theta0[i] - h
            model.set_params(th)
            j_minus = model.loss(X, y)
            fd[i] = (j_plus - j_minus) / (2.0 * h)
    finally:
        model.set_params(theta0)
    return fd


def gradient_check(model: MLP, X: np.ndarray, y: np.ndarray, h: float = H_FD) -> Dict[str, float]:
    """Compare backprop and finite-difference gradients at all parameters.

    Returns a dict with ``P``, ``max_abs_gap`` and ``max_rel_gap``
    (relative gap = |a - n| / max(|a|, |n|, 1e-12), reported for information only).
    """
    _, g = model.loss_and_grads(np.asarray(X, dtype=float).T, y)
    analytic = model.flatten_grads(g)
    numeric = finite_difference_gradient(model, X, y, h)
    abs_gap = np.abs(analytic - numeric)
    rel_gap = abs_gap / np.maximum(np.maximum(np.abs(analytic), np.abs(numeric)), 1e-12)
    return {
        "P": analytic.size,
        "max_abs_gap": float(abs_gap.max()),
        "max_rel_gap": float(rel_gap.max()),
    }


def lecture_example_model() -> tuple[MLP, np.ndarray, np.ndarray]:
    """2-2-1 sigmoid network of Ch 10 Part VIII with its initial parameters."""
    m = MLP([2, 2, 1], hidden_activation="sigmoid", lam=0.0, seed=0)
    m.W[0][...] = [[0.2, -0.3], [0.4, 0.1]]
    m.b[0][...] = [[0.0], [0.1]]
    m.W[1][...] = [[0.7, -0.5]]
    m.b[1][...] = [[0.2]]
    return m, np.array([[1.0, 0.0]]), np.array([1.0])


def main() -> int:
    """Run the check on the lecture example and on WDBC-sized tanh networks."""
    ok = True
    m, X, y = lecture_example_model()
    r = gradient_check(m, X, y)
    print(f"lecture 2-2-1 sigmoid, lam=0:        P={r['P']:4d}  max abs gap = {r['max_abs_gap']:.3e}")
    ok &= r["max_abs_gap"] < GAP_TOL

    rng = np.random.default_rng(123)
    Xr = rng.standard_normal((20, 30))
    yr = (rng.random(20) < 0.4).astype(float)
    for n1 in (4, 8, 16):
        for lam in (0.0, 1e-3, 1e-2):
            net = MLP([30, n1, 1], "tanh", lam=lam, init_scale=0.1, seed=0)
            r = gradient_check(net, Xr, yr)
            assert r["P"] == 32 * n1 + 1
            print(f"30-{n1:2d}-1 tanh, lam={lam:<6g}:        P={r['P']:4d}  max abs gap = {r['max_abs_gap']:.3e}")
            ok &= r["max_abs_gap"] < GAP_TOL
    print("PASS" if ok else "FAIL", f"(required max gap < {GAP_TOL:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
