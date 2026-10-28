"""From-scratch NumPy multilayer perceptron.

Notation follows the conventions ledger (U-03):
W^[l], b^[l], z^[l], a^[l], delta^[l]; a^[0] = x; vectors are columns.

Shapes
------
* ``W[l-1]`` stores W^[l] with shape (n_l, n_{l-1}); ``b[l-1]`` stores b^[l]
  with shape (n_l, 1).  (Python lists are 0-indexed, ledger layers are 1-indexed.)
* Internally a batch is a matrix ``A0`` of shape (n_0, N): one example per column.
* The public convenience methods ``fit``, ``predict_proba``, ``predict`` and
  ``loss`` take ``X`` of shape (N, d) (rows = examples, as in data_prep/sklearn)
  and ``y`` of shape (N,) with values in {0, 1}; they transpose internally.

Objective (U-19, U-20)
----------------------
J = (1/N_batch) sum_r BCE^(r) + (lambda/2) sum_l ||W^[l]||_F^2, biases not penalized.
BCE is computed from the logit z^[L] in a numerically stable form.

Only NumPy is used (no ML framework).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import INIT_SCALE

DEFAULT_INIT_SCALE = INIT_SCALE

# Elementwise functions
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, stable for large |z| (never evaluates exp of a large positive)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-example binary cross-entropy from the logit z = z^[L].

    BCE = -y log sigma(z) - (1-y) log(1 - sigma(z))
        = max(z, 0) - y z + log(1 + exp(-|z|)),
    which never overflows and never takes log(0).
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.maximum(z, 0.0) - y * z + np.log1p(np.exp(-np.abs(z)))


def _phi(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return sigmoid(z)
    raise ValueError(f"unknown hidden activation: {name!r}")


def _dphi_from_a(name: str, a: np.ndarray) -> np.ndarray:
    """phi'(z) expressed through a = phi(z)."""
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    raise ValueError(f"unknown hidden activation: {name!r}")

# Containers
@dataclass
class Cache:
    """Forward-pass quantities needed by backpropagation."""

    z: List[np.ndarray]  # z[l-1] = z^[l], l = 1..L
    a: List[np.ndarray]  # a[l]   = a^[l], l = 0..L  (a[0] = input batch)


@dataclass
class Grads:
    """Gradients of J and error signals delta^[l] (already divided by N_batch)."""

    dW: List[np.ndarray]
    db: List[np.ndarray]
    delta: List[np.ndarray]  # delta[l-1] = dJ_data/dz^[l], shape (n_l, N)


@dataclass
class History:
    """Per-epoch record of full-batch gradient descent.

    ``loss[k]`` and ``grad_norm[k]`` are J(theta_k) and ||grad J(theta_k)||_2
    evaluated at the parameters *before* update k+1; entry 0 is the
    initialization.  ``n_iter`` is the number of updates actually performed,
    so ``len(loss) == n_iter + 1``.
    """

    loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False  # True iff stopped by ||grad J||_2 < tol


# Model
class MLP:
    """Fully connected network, tanh (or sigmoid) hidden layers, sigmoid output, BCE.

    Parameters
    ----------
    layer_sizes : sequence of int
        (n_0, n_1, ..., n_L) with n_L = 1 for binary classification.
        For WDBC with one hidden layer: (30, n_1, 1), P = 32 n_1 + 1.
    hidden_activation : {"tanh", "sigmoid"}
        phi for hidden layers.  Project design is "tanh"; "sigmoid" exists only
        so the Ch 10 Part VIII lecture example (sigmoid hidden units) can be tested.
    lam : float
        Regularization strength lambda >= 0 (weights only, biases not penalized).
    init_scale : float
        Std of the zero-mean normal used for weights; biases start at 0.
    seed : int or None
        Seed of the initialization RNG (numpy ``default_rng``).
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activation: str = "tanh",
        lam: float = 0.0,
        init_scale: float = DEFAULT_INIT_SCALE,
        seed: Optional[int] = 0,
    ) -> None:
        sizes = list(layer_sizes)
        if len(sizes) < 2:
            raise ValueError("need at least input and output sizes")
        if sizes[-1] != 1:
            raise ValueError("binary MLP: output width n_L must be 1")
        if lam < 0:
            raise ValueError("lam must be >= 0")
        _phi(hidden_activation, np.zeros(1))  # validate name
        self.layer_sizes: List[int] = sizes
        self.hidden_activation = hidden_activation
        self.lam = float(lam)
        self.init_scale = float(init_scale)
        self.L = len(sizes) - 1
        rng = np.random.default_rng(seed)
        self.W: List[np.ndarray] = [
            init_scale * rng.standard_normal((sizes[l], sizes[l - 1]))
            for l in range(1, self.L + 1)
        ]
        self.b: List[np.ndarray] = [np.zeros((sizes[l], 1)) for l in range(1, self.L + 1)]

    #parameters
    @property
    def n_params(self) -> int:
        """P = sum_l n_l (n_{l-1} + 1)."""
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    def get_params(self) -> np.ndarray:
        """Flatten theta in the order W^[1], b^[1], W^[2], b^[2], ... (row-major)."""
        return np.concatenate(
            [p.ravel() for Wl, bl in zip(self.W, self.b) for p in (Wl, bl)]
        )

    def set_params(self, theta: np.ndarray) -> None:
        """Inverse of :meth:`get_params`."""
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {theta.size}")
        i = 0
        for l in range(self.L):
            for arr in (self.W[l], self.b[l]):
                arr[...] = theta[i : i + arr.size].reshape(arr.shape)
                i += arr.size

    @staticmethod
    def flatten_grads(g: Grads) -> np.ndarray:
        """Flatten gradients in the same order as :meth:`get_params`."""
        return np.concatenate(
            [p.ravel() for dWl, dbl in zip(g.dW, g.db) for p in (dWl, dbl)]
        )

    #forward / loss / backward
    def forward(self, A0: np.ndarray) -> Cache:
        """Forward pass on a batch A0 of shape (n_0, N) (one example per column)."""
        A0 = np.asarray(A0, dtype=float)
        if A0.ndim != 2 or A0.shape[0] != self.layer_sizes[0]:
            raise ValueError(f"A0 must have shape ({self.layer_sizes[0]}, N)")
        a: List[np.ndarray] = [A0]
        z: List[np.ndarray] = []
        for l in range(self.L):
            zl = self.W[l] @ a[-1] + self.b[l]
            z.append(zl)
            a.append(sigmoid(zl) if l == self.L - 1 else _phi(self.hidden_activation, zl))
        return Cache(z=z, a=a)

    def objective(self, cache: Cache, y: np.ndarray) -> float:
        """J = mean BCE + (lambda/2) sum ||W^[l]||_F^2 for a forward cache (y shape (N,) or (1,N))."""
        y = np.asarray(y, dtype=float).reshape(1, -1)
        data = float(np.mean(bce_with_logits(cache.z[-1], y)))
        reg = 0.5 * self.lam * sum(float(np.sum(W * W)) for W in self.W)
        return data + reg

    def backward(self, cache: Cache, y: np.ndarray) -> Grads:
        """Backpropagation (Ch 10 Thm 5.2 and Thm 7.1) for J including the regularizer.

        delta^[L] = (p_hat - y)/N;  delta^[l] = (W^[l+1]^T delta^[l+1]) (.) phi'(z^[l]);
        dW^[l] = delta^[l] a^[l-1]^T + lambda W^[l];  db^[l] = sum over examples of delta^[l].
        """
        y = np.asarray(y, dtype=float).reshape(1, -1)
        N = y.shape[1]
        delta: List[np.ndarray] = [np.empty((0, 0))] * self.L
        delta[-1] = (cache.a[-1] - y) / N
        for l in range(self.L - 2, -1, -1):  # layer l+1 in ledger indexing
            delta[l] = (self.W[l + 1].T @ delta[l + 1]) * _dphi_from_a(
                self.hidden_activation, cache.a[l + 1]
            )
        dW = [delta[l] @ cache.a[l].T + self.lam * self.W[l] for l in range(self.L)]
        db = [delta[l].sum(axis=1, keepdims=True) for l in range(self.L)]
        return Grads(dW=dW, db=db, delta=delta)

    def loss_and_grads(self, A0: np.ndarray, y: np.ndarray) -> Tuple[float, Grads]:
        """Return (J, gradients) for a batch in column layout (n_0, N)."""
        cache = self.forward(A0)
        return self.objective(cache, y), self.backward(cache, y)

    #public (N, d) convenience API
    def loss(self, X: np.ndarray, y: np.ndarray, include_reg: bool = True) -> float:
        """J on data X (N, d), y (N,).  ``include_reg=False`` gives the mean BCE only."""
        cache = self.forward(np.asarray(X, dtype=float).T)
        if include_reg:
            return self.objective(cache, y)
        y2 = np.asarray(y, dtype=float).reshape(1, -1)
        return float(np.mean(bce_with_logits(cache.z[-1], y2)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimated P(malignant) = p_hat for each row of X (N, d); shape (N,)."""
        return self.forward(np.asarray(X, dtype=float).T).a[-1].ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard labels in {0, 1}: p_hat >= threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eta: float,
        max_epochs: int,
        tol: float,
    ) -> History:
        """Full-batch gradient descent theta <- theta - eta * grad J.

        Stops when ||grad J(theta)||_2 < tol (checked before each update) or after
        ``max_epochs`` updates, whichever comes first.  For full-batch descent one
        epoch equals one iteration.  Records J and ||grad J||_2 at every visited
        theta, starting with the initialization.
        """
        if eta <= 0:
            raise ValueError("eta must be > 0")
        A0 = np.asarray(X, dtype=float).T
        hist = History()
        for epoch in range(max_epochs + 1):
            cache = self.forward(A0)
            J = self.objective(cache, y)
            g = self.backward(cache, y)
            gn = float(np.linalg.norm(self.flatten_grads(g)))
            hist.loss.append(J)
            hist.grad_norm.append(gn)
            if gn < tol:
                hist.converged = True
                break
            if epoch == max_epochs:
                break
            for l in range(self.L):
                self.W[l] -= eta * g.dW[l]
                self.b[l] -= eta * g.db[l]
            hist.n_iter += 1
        return hist

    def info(self) -> Dict[str, object]:
        """Small summary dict (widths, P, activation, lambda) for logging."""
        return {
            "layer_sizes": self.layer_sizes,
            "P": self.n_params,
            "hidden_activation": self.hidden_activation,
            "lam": self.lam,
            "init_scale": self.init_scale,
        }
