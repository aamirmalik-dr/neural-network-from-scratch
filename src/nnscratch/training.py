"""Minibatch training loop and metric tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nnscratch.losses import SoftmaxCrossEntropy
from nnscratch.model import Sequential
from nnscratch.optimizers import Optimizer


@dataclass
class History:
    """Per-epoch training metrics.

    Attributes:
        loss: Mean regularized training loss per epoch.
        train_accuracy: Training set accuracy per epoch.
        val_accuracy: Validation accuracy per epoch (empty if no validation
            data was provided).
    """

    loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def accuracy(model: Sequential, x: np.ndarray, y: np.ndarray) -> float:
    """Classification accuracy of the model on the given data.

    Args:
        model: Trained or in-training model.
        x: Inputs.
        y: Integer labels.

    Returns:
        Fraction of correct predictions.
    """
    return float((model.predict(x) == y).mean())


def train(
    model: Sequential,
    optimizer: Optimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> History:
    """Train a model with minibatch gradient descent.

    Args:
        model: Model to train.
        optimizer: Optimizer applied after each minibatch.
        x_train: Training inputs of shape (n, features).
        y_train: Integer training labels of shape (n,).
        x_val: Optional validation inputs.
        y_val: Optional validation labels.
        epochs: Number of passes over the training data.
        batch_size: Minibatch size.
        rng: Random generator for shuffling. A fresh default generator is used
            when omitted.
        verbose: Print per-epoch metrics when True.

    Returns:
        A History with per-epoch loss and accuracy.
    """
    if rng is None:
        rng = np.random.default_rng()
    loss_fn = SoftmaxCrossEntropy()
    history = History()
    n = x_train.shape[0]

    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            batch_loss = model.loss(xb, yb, loss_fn, training=True)
            model.backward(loss_fn.backward())
            optimizer.step(model.parameters(), model.gradients())
            epoch_loss += batch_loss
            batches += 1

        history.loss.append(epoch_loss / max(batches, 1))
        history.train_accuracy.append(accuracy(model, x_train, y_train))
        if x_val is not None and y_val is not None:
            history.val_accuracy.append(accuracy(model, x_val, y_val))

        if verbose:
            msg = (
                f"epoch {epoch + 1:3d}/{epochs}  "
                f"loss {history.loss[-1]:.4f}  "
                f"train_acc {history.train_accuracy[-1]:.4f}"
            )
            if history.val_accuracy:
                msg += f"  val_acc {history.val_accuracy[-1]:.4f}"
            print(msg)

    return history
