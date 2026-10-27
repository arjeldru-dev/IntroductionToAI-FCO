"""WDBC loading, encoding, stratified split, train-only standardization.

Run: python code/data_prep.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402


# ---------------------------------------------------------------- download
def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wdbc(path: Path = cfg.DATA_FILE) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the official UCI wdbc.data, verify checksum, encode malignant=1, benign=0.

    File format (see data/wdbc.names): id, diagnosis (M/B), then 30 real-valued features;
    no header. The ID column is dropped. Returns X (N x d), y (N,) int, feature names.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download wdbc.data from "
                                "https://archive.ics.uci.edu/dataset/17 into data/")
    expected = cfg.CHECKSUM_FILE.read_text().split()[0]
    if sha256_of(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate IDs")
    raw = [r[1] for r in rows]
    if set(raw) != {cfg.POSITIVE_LABEL, cfg.NEGATIVE_LABEL}:
        raise ValueError(f"unexpected labels: {set(raw)}")
    X = np.array([[float(v) for v in r[2:]] for r in rows])
    y = np.array([1 if v == cfg.POSITIVE_LABEL else 0 for v in raw], dtype=int)
    if X.shape != (cfg.N_SAMPLES, cfg.N_FEATURES):
        raise ValueError(f"unexpected shape {X.shape}")
    return X, y, list(cfg.FEATURE_NAMES)


# ---------------------------------------------------------------- split + scale
def stratified_split(y: np.ndarray, seed: int = cfg.SPLIT_SEED
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70/15/15 index split (train, val, test), deterministic in seed."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=cfg.VAL_FRAC + cfg.TEST_FRAC,
                                stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=cfg.TEST_FRAC / (cfg.VAL_FRAC + cfg.TEST_FRAC),
                              stratify=y[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


@dataclass(frozen=True)
class Standardizer:
    """z = (x - mu_train) / s_train; population std (ddof=0), fitted on TRAIN only."""
    mu_train: np.ndarray
    s_train: np.ndarray

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "Standardizer":
        s = X_train.std(axis=0, ddof=0)
        if np.any(s == 0):
            raise ValueError("zero-variance feature in training data")
        return cls(X_train.mean(axis=0), s)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_train) / self.s_train


@dataclass
class Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Standardizer
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    feature_names: list[str]


def prepare_data(seed: int = cfg.SPLIT_SEED) -> Data:
    """Full pipeline: load -> split -> fit scaler on train -> transform all splits."""
    X, y, names = load_wdbc()
    tr, va, te = stratified_split(y, seed)
    sc = Standardizer.fit(X[tr])
    return Data(sc.transform(X[tr]), y[tr], sc.transform(X[va]), y[va],
                sc.transform(X[te]), y[te], sc, tr, va, te, names)


def main() -> None:
    X, y, _ = load_wdbc()
    print(f"file: {cfg.DATA_FILE.relative_to(cfg.ROOT)}  sha256: {sha256_of(cfg.DATA_FILE)}")
    print(f"shape: {X.shape}  malignant(1)={int(y.sum())}  benign(0)={int((y == 0).sum())}")
    d = prepare_data()
    print(f"split seed {cfg.SPLIT_SEED}")
    for nm, yy, XX in (("train", d.y_train, d.X_train), ("val", d.y_val, d.X_val),
                       ("test", d.y_test, d.X_test)):
        print(f"  {nm:5s} n={len(yy):3d} malignant={int(yy.sum()):3d} benign={int((yy == 0).sum()):3d}"
              f"  malignant frac={yy.mean():.4f}  max|feature mean|={np.abs(XX.mean(0)).max():.3g}")


if __name__ == "__main__":
    main()
