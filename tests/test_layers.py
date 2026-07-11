"""Shape and behavior tests for individual layers."""

import numpy as np
import pytest

from nnscratch.initializers import get_initializer, he_init, xavier_init
from nnscratch.layers import Dense, Dropout, ReLU, Sigmoid
from nnscratch.losses import SoftmaxCrossEntropy


def test_dense_shapes():
    rng = np.random.default_rng(0)
    layer = Dense(5, 3, rng=rng)
    x = rng.standard_normal((7, 5))
    out = layer.forward(x)
    assert out.shape == (7, 3)
    grad_in = layer.backward(np.ones_like(out))
    assert grad_in.shape == x.shape
    assert layer.grad_w.shape == layer.w.shape
    assert layer.grad_b.shape == layer.b.shape


def test_dense_params_grads_aligned():
    layer = Dense(4, 2, rng=np.random.default_rng(0))
    x = np.random.default_rng(1).standard_normal((3, 4))
    layer.forward(x)
    layer.backward(np.ones((3, 2)))
    assert len(layer.params) == len(layer.grads) == 2
    for p, g in zip(layer.params, layer.grads, strict=True):
        assert p.shape == g.shape


def test_relu_forward_backward():
    layer = ReLU()
    x = np.array([[-1.0, 0.0, 2.0]])
    out = layer.forward(x)
    np.testing.assert_array_equal(out, [[0.0, 0.0, 2.0]])
    grad = layer.backward(np.array([[1.0, 1.0, 1.0]]))
    np.testing.assert_array_equal(grad, [[0.0, 0.0, 1.0]])


def test_sigmoid_range_and_stability():
    layer = Sigmoid()
    x = np.array([[-1000.0, 0.0, 1000.0]])
    out = layer.forward(x)
    assert np.all(np.isfinite(out))
    assert out[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert out[0, 1] == pytest.approx(0.5)
    assert out[0, 2] == pytest.approx(1.0, abs=1e-12)


def test_dropout_inference_is_identity():
    layer = Dropout(0.5, rng=np.random.default_rng(0))
    x = np.random.default_rng(1).standard_normal((10, 20))
    np.testing.assert_array_equal(layer.forward(x, training=False), x)


def test_dropout_training_scales_survivors():
    rate = 0.4
    layer = Dropout(rate, rng=np.random.default_rng(0))
    x = np.ones((200, 200))
    out = layer.forward(x, training=True)
    dropped = float((out == 0.0).mean())
    assert dropped == pytest.approx(rate, abs=0.02)
    survivors = out[out != 0.0]
    np.testing.assert_allclose(survivors, 1.0 / (1.0 - rate))
    # Inverted scaling keeps the expected activation roughly unchanged.
    assert out.mean() == pytest.approx(1.0, abs=0.02)


def test_dropout_rejects_bad_rate():
    with pytest.raises(ValueError):
        Dropout(1.0)


def test_softmax_cross_entropy_known_value():
    loss_fn = SoftmaxCrossEntropy()
    logits = np.zeros((2, 4))
    labels = np.array([0, 3])
    loss = loss_fn.forward(logits, labels)
    assert loss == pytest.approx(np.log(4.0))
    grad = loss_fn.backward()
    assert grad.shape == (2, 4)
    # Gradient rows sum to zero for softmax cross entropy.
    np.testing.assert_allclose(grad.sum(axis=1), 0.0, atol=1e-12)


def test_initializer_scales():
    rng = np.random.default_rng(0)
    w_he = he_init(1000, 50, rng)
    assert w_he.std() == pytest.approx(np.sqrt(2.0 / 1000), rel=0.1)
    w_xavier = xavier_init(1000, 1000, rng)
    assert w_xavier.std() == pytest.approx(np.sqrt(2.0 / 2000), rel=0.1)


def test_get_initializer_unknown_name():
    with pytest.raises(ValueError):
        get_initializer("lecun")
