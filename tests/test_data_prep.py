"""Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as cfg  # noqa: E402
import data_prep as dp  # noqa: E402


def test_shape_and_encoding():
    X, y, names = dp.load_wdbc()
    assert X.shape == (569, 30) and len(names) == 30
    assert set(np.unique(y)) == {0, 1}


def test_checksum_matches():
    assert dp.sha256_of(cfg.DATA_FILE) == cfg.CHECKSUM_FILE.read_text().split()[0]


def test_split_disjoint_complete_stratified():
    _, y, _ = dp.load_wdbc()
    tr, va, te = dp.stratified_split(y)
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == 569 and len(set(allidx)) == 569
    assert abs(len(tr) / 569 - 0.70) < 0.01
    for s in (tr, va, te):
        assert abs(y[s].mean() - y.mean()) < 0.02


def test_deterministic():
    a, b = dp.prepare_data(), dp.prepare_data()
    assert np.array_equal(a.idx_val, b.idx_val) and np.array_equal(a.X_train, b.X_train)


def test_scaler_fitted_on_train_only():
    X, y, _ = dp.load_wdbc()
    d = dp.prepare_data()
    # (1) scaler stats equal train-only statistics, and differ from full-data statistics
    assert np.allclose(d.scaler.mu_train, X[d.idx_train].mean(0))
    assert np.allclose(d.scaler.s_train, X[d.idx_train].std(0, ddof=0))
    assert not np.allclose(d.scaler.mu_train, X.mean(0))
    # (2) train is exactly standardized; val is NOT (it would be if fitted on it)
    assert np.allclose(d.X_train.mean(0), 0, atol=1e-10) and np.allclose(d.X_train.std(0), 1)
    assert not np.allclose(d.X_val.mean(0), 0, atol=1e-3)
    # (3) perturbing val/test raw data must not change the fitted scaler
    X2 = X.copy(); X2[d.idx_val] *= 100; X2[d.idx_test] += 50
    sc2 = dp.Standardizer.fit(X2[d.idx_train])
    assert np.array_equal(sc2.mu_train, d.scaler.mu_train)
