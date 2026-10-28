"""Reproduce the Ch 10 Part VIII 2-2-1 sigmoid example with the group's MLP (6 decimals),
plus the finite-difference gradient check on it.  Logic mirrors verify_ch10_example.py.

Run:  pytest tests/test_lecture_example.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from gradient_check import gradient_check, lecture_example_model  # noqa: E402

ETA = 0.1
ATOL = 5e-7  # agreement to 6 decimals


@pytest.fixture()
def example():
    return lecture_example_model()


def test_forward_values(example):
    m, X, y = example
    cache = m.forward(X.T)
    assert np.allclose(cache.a[1].ravel(), [0.549834, 0.622459], atol=ATOL)
    assert np.allclose(cache.a[2].item(), 0.567990, atol=ATOL)
    assert np.allclose(m.objective(cache, y), 0.565652, atol=ATOL)


def test_backward_values(example):
    m, X, y = example
    cache = m.forward(X.T)
    g = m.backward(cache, y)
    assert np.allclose(g.delta[1].item(), -0.432010, atol=ATOL)
    assert np.allclose(g.delta[0].ravel(), [-0.074851, 0.050762], atol=ATOL)
    assert np.allclose(g.dW[1].ravel(), [-0.237534, -0.268909], atol=ATOL)


def test_one_gradient_step(example):
    m, X, y = example
    hist = m.fit(X, y, eta=ETA, max_epochs=1, tol=0.0)
    assert hist.n_iter == 1
    assert np.allclose(m.predict_proba(X).item(), 0.586730, atol=ATOL)
    assert np.allclose(m.loss(X, y), 0.533190, atol=ATOL)
    assert np.allclose(hist.loss[0], 0.565652, atol=ATOL)


def test_gradient_check_lecture_example(example):
    m, X, y = example
    r = gradient_check(m, X, y, h=1e-5)
    assert r["P"] == 9
    assert r["max_abs_gap"] < 1e-6
