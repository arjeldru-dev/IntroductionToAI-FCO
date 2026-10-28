"""Independent check of the worked example in MCS 306 Chapter 10

Network: 2-2-1, sigmoid hidden and output units, binary cross-entropy,
one training example x = (1, 0)^T, y = 1, learning rate eta = 0.1.

Compares our NumPy forward/backward pass against the values printed in the
lecture notes (6 decimal places) and against a centered finite-difference
gradient check (h = 1e-5).

Intended later as tests/test_lecture_example.py for the MLP code.
Run:  python verify_ch10_example.py
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


X = np.array([[1.0], [0.0]])
Y = 1.0
W1 = np.array([[0.2, -0.3], [0.4, 0.1]])
B1 = np.array([[0.0], [0.1]])
W2 = np.array([[0.7, -0.5]])
B2 = np.array([[0.2]])
ETA = 0.1


def forward(W1, B1, W2, B2):
    z1 = W1 @ X + B1
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + B2
    p = sigmoid(z2)
    return z1, a1, z2, p


def bce(p):
    return -np.log(p.item())  # y = 1


z1, a1, z2, p = forward(W1, B1, W2, B2)
J = bce(p)

# Backward pass
d2 = p - Y                                   # delta^[2]
dW2 = d2 @ a1.T
db2 = d2
d1 = (W2.T @ d2) * a1 * (1.0 - a1)           # delta^[1]
dW1 = d1 @ X.T
db1 = d1

# One gradient-descent step
W1n, B1n = W1 - ETA * dW1, B1 - ETA * db1
W2n, B2n = W2 - ETA * dW2, B2 - ETA * db2
p_new = forward(W1n, B1n, W2n, B2n)[3]
J_new = bce(p_new)

# Values quoted in the lecture notes
expected = {
    "a1": (a1.ravel(), [0.549834, 0.622459]),
    "p": (p.item(), 0.567990),
    "J": (J, 0.565652),
    "delta2": (d2.item(), -0.432010),
    "delta1": (d1.ravel(), [-0.074851, 0.050762]),
    "dW2": (dW2.ravel(), [-0.237534, -0.268909]),
    "p_new": (p_new.item(), 0.586730),
    "J_new": (J_new, 0.533190),
}
for name, (got, want) in expected.items():
    assert np.allclose(got, want, atol=5e-7), (name, got, want)
    print(f"OK {name}: {np.round(got, 6)}")


# Centered finite-difference check over all 9 parameters
def loss_from_theta(th):
    w1 = th[0:4].reshape(2, 2)
    b1 = th[4:6].reshape(2, 1)
    w2 = th[6:8].reshape(1, 2)
    b2 = th[8:9].reshape(1, 1)
    return bce(forward(w1, b1, w2, b2)[3])


theta = np.concatenate([W1.ravel(), B1.ravel(), W2.ravel(), B2.ravel()])
grad = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
h = 1e-5
fd = np.array([
    (loss_from_theta(theta + h * e) - loss_from_theta(theta - h * e)) / (2 * h)
    for e in np.eye(9)
])
gap = np.abs(fd - grad).max()
print("max |analytic - finite difference| =", gap)
assert gap < 1e-8
print("All checks passed.")
