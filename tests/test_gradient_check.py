"""Numerical gradient checks: the correctness proof for backpropagation.

Two levels of checking:

1. Per-layer checks. For each layer, the gradient of a scalar probe function
   ``f = sum(forward(x) * r)`` with respect to the layer input (and, for
   Dense, its weights and bias) is compared entry by entry against central
   difference estimates.
2. Whole-network checks. ``gradient_check`` compares the full
   backpropagation pass against numerical gradients of the regularized loss
   for several architectures, activations, and L2 settings.
"""

import numpy as np

from nnscratch.gradient_check import gradient_check
from nnscratch.layers import Dense, ReLU, Sigmoid
from nnscratch.losses import SoftmaxCrossEntropy
from nnscratch.model import Sequential

TOLERANCE = 1e-5
EPS = 1e-6


def _random_batch(rng, n=8, features=6, classes=4):
    x = rng.standard_normal((n, features))
    y = rng.integers(0, classes, size=n)
    return x, y


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a) + np.linalg.norm(b)) + 1e-12
    return num / den


def _numerical_grad(f, arr: np.ndarray) -> np.ndarray:
    """Central-difference gradient of scalar f with respect to arr, in place."""
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = arr[idx]
        arr[idx] = original + EPS
        f_plus = f()
        arr[idx] = original - EPS
        f_minus = f()
        arr[idx] = original
        grad[idx] = (f_plus - f_minus) / (2.0 * EPS)
        it.iternext()
    return grad


def _check_input_gradient(layer, x: np.ndarray, rng) -> float:
    """Compare layer.backward against numerical d(sum(out * r))/dx."""
    r = rng.standard_normal(layer.forward(x).shape)
    analytic = layer.backward(r)
    numeric = _numerical_grad(lambda: float((layer.forward(x) * r).sum()), x)
    return _rel_error(analytic, numeric)


def test_dense_input_weight_bias_gradients():
    rng = np.random.default_rng(0)
    layer = Dense(5, 3, init="he", rng=rng)
    x = rng.standard_normal((4, 5))
    r = rng.standard_normal((4, 3))

    layer.forward(x)
    analytic_x = layer.backward(r)
    analytic_w = layer.grad_w.copy()
    analytic_b = layer.grad_b.copy()

    def probe() -> float:
        return float((layer.forward(x) * r).sum())

    assert _rel_error(analytic_x, _numerical_grad(probe, x)) < TOLERANCE
    assert _rel_error(analytic_w, _numerical_grad(probe, layer.w)) < TOLERANCE
    assert _rel_error(analytic_b, _numerical_grad(probe, layer.b)) < TOLERANCE


def test_dense_l2_term_matches_penalty_gradient():
    # With L2 active, grad_w must equal the data term plus l2 * w, and the
    # l2_penalty must be the function whose gradient is that extra term.
    rng = np.random.default_rng(1)
    layer = Dense(4, 3, init="he", l2=0.1, rng=rng)
    x = rng.standard_normal((6, 4))
    r = rng.standard_normal((6, 3))
    layer.forward(x)
    layer.backward(r)
    analytic_w = layer.grad_w.copy()

    def probe() -> float:
        return float((layer.forward(x) * r).sum()) + layer.l2_penalty()

    assert _rel_error(analytic_w, _numerical_grad(probe, layer.w)) < TOLERANCE


def test_relu_input_gradient():
    rng = np.random.default_rng(2)
    layer = ReLU()
    # Keep inputs away from the kink at 0 where the derivative is undefined.
    x = rng.standard_normal((5, 7))
    x = np.where(np.abs(x) < 0.1, x + 0.2, x)
    assert _check_input_gradient(layer, x, rng) < TOLERANCE


def test_sigmoid_input_gradient():
    rng = np.random.default_rng(3)
    layer = Sigmoid()
    x = rng.standard_normal((5, 7)) * 3.0
    assert _check_input_gradient(layer, x, rng) < TOLERANCE


def test_softmax_cross_entropy_gradient():
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((6, 5))
    labels = rng.integers(0, 5, size=6)
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(logits, labels)
    analytic = loss_fn.backward()
    numeric = _numerical_grad(lambda: loss_fn.forward(logits, labels), logits)
    assert _rel_error(analytic, numeric) < TOLERANCE


def test_gradient_check_relu_network():
    rng = np.random.default_rng(42)
    x, y = _random_batch(rng)
    model = Sequential(
        [
            Dense(6, 10, init="he", rng=rng),
            ReLU(),
            Dense(10, 4, init="xavier", rng=rng),
        ]
    )
    rel_error = gradient_check(model, SoftmaxCrossEntropy(), x, y)
    assert rel_error < TOLERANCE


def test_gradient_check_sigmoid_network():
    rng = np.random.default_rng(7)
    x, y = _random_batch(rng)
    model = Sequential(
        [
            Dense(6, 8, init="xavier", rng=rng),
            Sigmoid(),
            Dense(8, 4, init="xavier", rng=rng),
        ]
    )
    rel_error = gradient_check(model, SoftmaxCrossEntropy(), x, y)
    assert rel_error < TOLERANCE


def test_gradient_check_with_l2():
    rng = np.random.default_rng(3)
    x, y = _random_batch(rng)
    model = Sequential(
        [
            Dense(6, 10, init="he", l2=0.05, rng=rng),
            ReLU(),
            Dense(10, 4, init="xavier", l2=0.01, rng=rng),
        ]
    )
    rel_error = gradient_check(model, SoftmaxCrossEntropy(), x, y)
    assert rel_error < TOLERANCE


def test_gradient_check_deeper_network():
    rng = np.random.default_rng(11)
    x, y = _random_batch(rng)
    model = Sequential(
        [
            Dense(6, 8, init="he", rng=rng),
            ReLU(),
            Dense(8, 8, init="he", l2=0.02, rng=rng),
            Sigmoid(),
            Dense(8, 4, init="xavier", rng=rng),
        ]
    )
    rel_error = gradient_check(model, SoftmaxCrossEntropy(), x, y)
    assert rel_error < TOLERANCE
