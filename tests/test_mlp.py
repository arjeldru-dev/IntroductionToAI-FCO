"""Unit tests for code/mlp.py: parameter count, stable BCE, regularizer, stopping rule,
reproducibility, gradient check with lambda > 0, and a tiny overfit sanity test.

Run:  pytest tests/test_mlp.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from gradient_check import gradient_check  # noqa: E402
from mlp import MLP, bce_with_logits, sigmoid  # noqa: E402


def _toy(n=20, d=30, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (rng.random(n) < 0.5).astype(float)
    return X, y


@pytest.mark.parametrize("n1", [4, 8, 16])
def test_parameter_count(n1):
    assert MLP([30, n1, 1]).n_params == 32 * n1 + 1


def test_biases_start_at_zero_and_seed_reproducible():
    a, b, c = MLP([30, 4, 1], seed=3), MLP([30, 4, 1], seed=3), MLP([30, 4, 1], seed=4)
    assert all(np.all(bl == 0) for bl in a.b)
    assert np.array_equal(a.get_params(), b.get_params())
    assert not np.array_equal(a.get_params(), c.get_params())


def test_stable_bce_extreme_logits():
    z = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
    for yv in (0.0, 1.0):
        out = bce_with_logits(z, np.full_like(z, yv))
        assert np.all(np.isfinite(out)) and np.all(out >= 0)
    assert np.isclose(bce_with_logits(np.array([0.0]), np.array([1.0])).item(), np.log(2))
    assert np.isclose(bce_with_logits(np.array([800.0]), np.array([0.0])).item(), 800.0)
    assert np.all(np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))))


def test_bce_matches_naive_formula_in_safe_range():
    z = np.linspace(-5, 5, 11)
    for yv in (0.0, 1.0):
        p = 1 / (1 + np.exp(-z))
        naive = -(yv * np.log(p) + (1 - yv) * np.log(1 - p))
        assert np.allclose(bce_with_logits(z, np.full_like(z, yv)), naive, atol=1e-12)


def test_regularizer_excludes_biases_and_adds_lambda_W():
    X, y = _toy(10)
    m0 = MLP([30, 4, 1], lam=0.0, seed=0)
    m1 = MLP([30, 4, 1], lam=0.01, seed=0)
    m0.b[0][...] = 0.7  # nonzero biases must not enter the penalty
    m1.b[0][...] = 0.7
    expected = 0.5 * 0.01 * sum(np.sum(W * W) for W in m1.W)
    assert np.isclose(m1.loss(X, y) - m0.loss(X, y), expected)
    _, g0 = m0.loss_and_grads(X.T, y)
    _, g1 = m1.loss_and_grads(X.T, y)
    for l in range(2):
        assert np.allclose(g1.dW[l] - g0.dW[l], 0.01 * m1.W[l])
        assert np.allclose(g1.db[l], g0.db[l])


@pytest.mark.parametrize("lam", [0.0, 1e-2])
def test_gradient_check_tanh_all_parameters(lam):
    X, y = _toy(15)
    m = MLP([30, 8, 1], "tanh", lam=lam, seed=2)
    r = gradient_check(m, X, y)
    assert r["P"] == 257
    assert r["max_abs_gap"] < 1e-6


def test_get_set_params_roundtrip():
    m = MLP([30, 4, 1], seed=5)
    th = m.get_params()
    m.set_params(th + 1.0)
    assert np.allclose(m.get_params(), th + 1.0)


def test_stopping_rule_and_history_lengths():
    X, y = _toy(10)
    m = MLP([30, 4, 1], seed=0)
    h = m.fit(X, y, eta=0.1, max_epochs=7, tol=0.0)  # never converges
    assert h.n_iter == 7 and not h.converged
    assert len(h.loss) == len(h.grad_norm) == 8
    m2 = MLP([30, 4, 1], seed=0)
    h2 = m2.fit(X, y, eta=0.1, max_epochs=5000, tol=1e9)  # stops at once
    assert h2.converged and h2.n_iter == 0 and len(h2.loss) == 1


def test_overfit_tiny_subset_loss_decreases():
    X, y = _toy(16, seed=7)
    m = MLP([30, 8, 1], "tanh", lam=0.0, init_scale=0.1, seed=0)
    h = m.fit(X, y, eta=0.5, max_epochs=2000, tol=1e-6)
    assert h.loss[-1] < 0.1 * h.loss[0]
    assert all(b <= a + 1e-12 for a, b in zip(h.loss, h.loss[1:]))  # monotone for small eta
    assert (m.predict(X) == y).all()


def test_overfit_small_real_wdbc_subset():
    """Optional: 40 standardized samples from sklearn's bundled WDBC copy (no test-set use)."""
    datasets = pytest.importorskip("sklearn.datasets")
    d = datasets.load_breast_cancer()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d.target), 40, replace=False)
    X = d.data[idx]
    X = (X - X.mean(0)) / X.std(0)
    y = (d.target[idx] == 0).astype(float)  # sklearn: 0 = malignant -> label 1 = malignant
    m = MLP([30, 4, 1], "tanh", lam=0.0, seed=0)
    h = m.fit(X, y, eta=0.1, max_epochs=1500, tol=1e-6)
    assert h.loss[-1] < 0.5 * h.loss[0]
