"""Optimizer behavior tests."""

import numpy as np
import pytest

from nnscratch.layers import Dense, ReLU
from nnscratch.losses import SoftmaxCrossEntropy
from nnscratch.model import Sequential
from nnscratch.optimizers import SGD, Adam, Momentum, get_optimizer


def _toy_problem(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((64, 5))
    # Labels depend linearly on the inputs, so the problem is learnable.
    w_true = rng.standard_normal((5, 3))
    y = (x @ w_true).argmax(axis=1)
    model = Sequential(
        [
            Dense(5, 16, init="he", rng=rng),
            ReLU(),
            Dense(16, 3, init="xavier", rng=rng),
        ]
    )
    return model, x, y


@pytest.mark.parametrize(
    "optimizer",
    [SGD(lr=0.1), Momentum(lr=0.05, mu=0.9), Adam(lr=0.01)],
    ids=["sgd", "momentum", "adam"],
)
def test_optimizer_reduces_loss(optimizer):
    model, x, y = _toy_problem()
    loss_fn = SoftmaxCrossEntropy()
    initial_loss = model.loss(x, y, loss_fn)
    for _ in range(50):
        model.loss(x, y, loss_fn, training=True)
        model.backward(loss_fn.backward())
        optimizer.step(model.parameters(), model.gradients())
    final_loss = model.loss(x, y, loss_fn)
    assert final_loss < 0.5 * initial_loss


def test_sgd_matches_manual_update():
    p = np.array([1.0, 2.0])
    g = np.array([0.5, -0.5])
    SGD(lr=0.1).step([p], [g])
    np.testing.assert_allclose(p, [0.95, 2.05])


def test_adam_first_step_size_is_lr():
    # With bias correction, the first Adam step has magnitude close to lr
    # regardless of gradient scale.
    p = np.array([0.0])
    g = np.array([1000.0])
    Adam(lr=0.001).step([p], [g])
    assert abs(p[0]) == pytest.approx(0.001, rel=1e-3)


def test_momentum_accumulates_velocity():
    p = np.array([0.0])
    g = np.array([1.0])
    opt = Momentum(lr=0.1, mu=0.9)
    opt.step([p], [g])
    first = p.copy()
    opt.step([p], [g])
    second_step = p - first
    # The second step is larger because velocity accumulates.
    assert abs(second_step[0]) > abs(first[0])


def test_get_optimizer_factory():
    assert isinstance(get_optimizer("sgd", lr=0.1), SGD)
    assert isinstance(get_optimizer("momentum", lr=0.1), Momentum)
    assert isinstance(get_optimizer("adam", lr=0.1), Adam)
    with pytest.raises(ValueError):
        get_optimizer("rmsprop", lr=0.1)
