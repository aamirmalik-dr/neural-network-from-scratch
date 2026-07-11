"""Numerical gradient checks for the full backpropagation stack."""

import numpy as np

from nnscratch.gradient_check import gradient_check
from nnscratch.layers import Dense, ReLU, Sigmoid
from nnscratch.losses import SoftmaxCrossEntropy
from nnscratch.model import Sequential

TOLERANCE = 1e-5


def _random_batch(rng, n=8, features=6, classes=4):
    x = rng.standard_normal((n, features))
    y = rng.integers(0, classes, size=n)
    return x, y


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
