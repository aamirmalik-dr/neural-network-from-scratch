"""End-to-end training tests on offline datasets, plus serialization."""

from pathlib import Path

import numpy as np
import pytest

from nnscratch.data import load_digits_data, load_two_moons, make_two_moons
from nnscratch.layers import Dense, Dropout, ReLU
from nnscratch.model import Sequential
from nnscratch.optimizers import Adam
from nnscratch.training import accuracy, train

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build(rng, in_features, n_classes, dropout=0.0):
    layers = [Dense(in_features, 64, init="he", l2=1e-4, rng=rng), ReLU()]
    if dropout > 0:
        layers.append(Dropout(dropout, rng=rng))
    layers.append(Dense(64, n_classes, init="xavier", rng=rng))
    return Sequential(layers)


def test_train_digits_reaches_high_accuracy():
    x_train, x_test, y_train, y_test = load_digits_data(seed=0)
    rng = np.random.default_rng(0)
    model = _build(rng, x_train.shape[1], 10)
    history = train(
        model,
        Adam(lr=0.001),
        x_train,
        y_train,
        x_val=x_test,
        y_val=y_test,
        epochs=15,
        batch_size=64,
        rng=rng,
    )
    assert history.loss[-1] < history.loss[0]
    assert accuracy(model, x_test, y_test) > 0.93
    assert len(history.loss) == len(history.train_accuracy) == len(history.val_accuracy) == 15


def test_train_with_dropout_still_learns():
    x_train, x_test, y_train, y_test = load_digits_data(seed=0)
    rng = np.random.default_rng(1)
    model = _build(rng, x_train.shape[1], 10, dropout=0.25)
    train(
        model,
        Adam(lr=0.001),
        x_train,
        y_train,
        epochs=15,
        batch_size=64,
        rng=rng,
    )
    assert accuracy(model, x_test, y_test) > 0.90


def test_make_two_moons_is_reproducible():
    x1, y1 = make_two_moons(200, noise=0.2, rng=np.random.default_rng(0))
    x2, y2 = make_two_moons(200, noise=0.2, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert x1.shape == (200, 2)
    assert set(np.unique(y1)) == {0, 1}
    assert y1.sum() == 100


def test_committed_two_moons_is_learnable():
    x_train, x_test, y_train, y_test = load_two_moons(
        path=REPO_ROOT / "data" / "two_moons.csv", seed=0
    )
    rng = np.random.default_rng(0)
    model = Sequential(
        [
            Dense(2, 32, init="he", rng=rng),
            ReLU(),
            Dense(32, 2, init="xavier", rng=rng),
        ]
    )
    train(model, Adam(lr=0.01), x_train, y_train, epochs=60, batch_size=32, rng=rng)
    assert accuracy(model, x_test, y_test) > 0.9


def test_save_and_load_weights_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    model = _build(rng, 8, 3)
    x = rng.standard_normal((4, 8))
    reference = model.forward(x)

    path = tmp_path / "weights.npz"
    model.save_weights(path)

    fresh = _build(np.random.default_rng(99), 8, 3)
    assert not np.allclose(fresh.forward(x), reference)
    fresh.load_weights(path)
    np.testing.assert_allclose(fresh.forward(x), reference)


def test_load_weights_rejects_wrong_architecture(tmp_path):
    rng = np.random.default_rng(0)
    model = _build(rng, 8, 3)
    path = tmp_path / "weights.npz"
    model.save_weights(path)
    other = _build(np.random.default_rng(0), 12, 3)
    with pytest.raises(ValueError):
        other.load_weights(path)


def test_model_forward_shapes():
    rng = np.random.default_rng(0)
    model = _build(rng, 64, 10)
    x = rng.standard_normal((5, 64))
    logits = model.forward(x)
    assert logits.shape == (5, 10)
    preds = model.predict(x)
    assert preds.shape == (5,)
    assert preds.dtype.kind == "i"
