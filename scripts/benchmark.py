"""Compare SGD, Momentum, and Adam on the same architecture and dataset.

Trains one model per optimizer from identical initial weights, saves a
convergence figure, and prints a final accuracy table.

Usage:
    python scripts/benchmark.py --dataset digits --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nnscratch.data import load_dataset
from nnscratch.layers import Dense, ReLU
from nnscratch.model import Sequential
from nnscratch.optimizers import get_optimizer
from nnscratch.training import History, accuracy, train


def build_model(in_features: int, hidden: int, n_classes: int, seed: int) -> Sequential:
    """Build the benchmark architecture with a fixed seed.

    Args:
        in_features: Input dimensionality.
        hidden: Hidden layer width.
        n_classes: Number of output classes.
        seed: Seed so every optimizer starts from identical weights.

    Returns:
        The assembled Sequential model.
    """
    rng = np.random.default_rng(seed)
    return Sequential(
        [
            Dense(in_features, hidden, init="he", l2=1e-4, rng=rng),
            ReLU(),
            Dense(hidden, hidden, init="he", l2=1e-4, rng=rng),
            ReLU(),
            Dense(hidden, n_classes, init="xavier", l2=1e-4, rng=rng),
        ]
    )


def main() -> None:
    """Run the optimizer comparison and save the convergence figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["digits", "mnist"], default="digits")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="figures/optimizer_comparison.png")
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = load_dataset(
        args.dataset, data_dir=args.data_dir, seed=args.seed
    )
    configs = [("sgd", 0.1), ("momentum", 0.05), ("adam", 0.001)]

    histories: dict[str, History] = {}
    final_acc: dict[str, float] = {}
    for name, lr in configs:
        model = build_model(x_train.shape[1], args.hidden, int(y_train.max()) + 1, args.seed)
        optimizer = get_optimizer(name, lr=lr)
        histories[name] = train(
            model,
            optimizer,
            x_train,
            y_train,
            x_val=x_test,
            y_val=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            rng=np.random.default_rng(args.seed),
        )
        final_acc[name] = accuracy(model, x_test, y_test)
        print(f"{name:9s} lr={lr:<6} final test accuracy: {final_acc[name]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, lr in configs:
        label = f"{name} (lr={lr})"
        epochs_axis = range(1, args.epochs + 1)
        ax1.plot(epochs_axis, histories[name].loss, label=label)
        ax2.plot(epochs_axis, histories[name].val_accuracy, label=label)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss")
    ax1.set_yscale("log")
    ax1.set_title(f"Convergence on {args.dataset}")
    ax1.legend()
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test accuracy")
    ax2.set_title("Held-out accuracy")
    ax2.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")

    print()
    print(f"{'optimizer':<12}{'lr':<8}{'test accuracy':<14}")
    for name, lr in configs:
        print(f"{name:<12}{lr:<8}{final_acc[name]:<14.4f}")


if __name__ == "__main__":
    main()
