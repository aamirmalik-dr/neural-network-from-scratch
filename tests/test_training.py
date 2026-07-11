"""End-to-end training tests on the bundled digits dataset."""

import numpy as np

from nnscratch.data import load_digits_data
from nnscratch.layers import Dense, Dropout, ReLU
from nnscratch.model import Sequential
from nnscratch.optimizers import Adam
from nnscratch.training import accuracy, train


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


def test_model_forward_shapes():
    rng = np.random.default_rng(0)
    model = _build(rng, 64, 10)
    x = rng.standard_normal((5, 64))
    logits = model.forward(x)
    assert logits.shape == (5, 10)
    preds = model.predict(x)
    assert preds.shape == (5,)
    assert preds.dtype.kind == "i"
