"""Train a feedforward network on two moons, digits, or MNIST from the CLI.

Usage:
    python scripts/train.py --dataset digits --optimizer adam --epochs 30
"""

from __future__ import annotations

import argparse

import numpy as np

from nnscratch.data import load_dataset
from nnscratch.layers import Dense, Dropout, ReLU
from nnscratch.model import Sequential
from nnscratch.optimizers import get_optimizer
from nnscratch.training import accuracy, train


def build_model(
    in_features: int,
    hidden: int,
    n_classes: int,
    l2: float,
    dropout: float,
    seed: int,
) -> Sequential:
    """Build a two hidden layer ReLU classifier.

    Args:
        in_features: Input dimensionality.
        hidden: Hidden layer width.
        n_classes: Number of output classes.
        l2: L2 regularization strength for all Dense layers.
        dropout: Dropout rate after each hidden activation (0 disables).
        seed: Seed for weight initialization and dropout masks.

    Returns:
        The assembled Sequential model.
    """
    rng = np.random.default_rng(seed)
    layers: list = [
        Dense(in_features, hidden, init="he", l2=l2, rng=rng),
        ReLU(),
    ]
    if dropout > 0:
        layers.append(Dropout(dropout, rng=rng))
    layers += [
        Dense(hidden, hidden, init="he", l2=l2, rng=rng),
        ReLU(),
    ]
    if dropout > 0:
        layers.append(Dropout(dropout, rng=rng))
    layers.append(Dense(hidden, n_classes, init="xavier", l2=l2, rng=rng))
    return Sequential(layers)


def main() -> None:
    """Parse arguments, train, and report final test accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["moons", "digits", "mnist"], default="digits")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="adam")
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (per-optimizer default if omitted)"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    default_lr = {"sgd": 0.1, "momentum": 0.05, "adam": 0.001}
    lr = args.lr if args.lr is not None else default_lr[args.optimizer]

    x_train, x_test, y_train, y_test = load_dataset(
        args.dataset, data_dir=args.data_dir, seed=args.seed
    )
    print(
        f"dataset={args.dataset} train={x_train.shape} test={x_test.shape} "
        f"optimizer={args.optimizer} lr={lr}"
    )

    model = build_model(
        in_features=x_train.shape[1],
        hidden=args.hidden,
        n_classes=int(y_train.max()) + 1,
        l2=args.l2,
        dropout=args.dropout,
        seed=args.seed,
    )
    optimizer = get_optimizer(args.optimizer, lr=lr)
    train(
        model,
        optimizer,
        x_train,
        y_train,
        x_val=x_test,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rng=np.random.default_rng(args.seed),
        verbose=True,
    )
    print(f"final test accuracy: {accuracy(model, x_test, y_test):.4f}")


if __name__ == "__main__":
    main()
