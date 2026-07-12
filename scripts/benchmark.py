"""Compare SGD, Momentum, and Adam on the same architecture and dataset.

Trains one model per optimizer from identical initial weights, saves a
convergence figure, and prints a final accuracy table.

Usage:
    python scripts/benchmark.py --dataset digits --epochs 40
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--out", default="results/optimizer_comparison.png")
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

    surface = "#fcfcfb"
    grid_color = "#e1e0d9"
    ink_muted = "#898781"
    series_colors = ("#2a78d6", "#1baf7a", "#eda100")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(surface)
    for (name, lr), color in zip(configs, series_colors, strict=True):
        label = f"{name} (lr={lr})"
        epochs_axis = range(1, args.epochs + 1)
        ax1.plot(epochs_axis, histories[name].loss, label=label, color=color, linewidth=2)
        ax2.plot(epochs_axis, histories[name].val_accuracy, label=label, color=color, linewidth=2)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid_color, linewidth=0.6)
        ax.tick_params(colors=ink_muted)
        ax.set_xlabel("epoch", color=ink_muted)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.legend()
    ax1.set_ylabel("training loss", color=ink_muted)
    ax1.set_yscale("log")
    ax1.set_title(f"Convergence on {args.dataset}")
    ax2.set_ylabel("test accuracy", color=ink_muted)
    ax2.set_title("Held-out accuracy")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=surface)
    print(f"figure saved to {out}")

    metrics = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed": args.seed,
        "architecture": f"{x_train.shape[1]}-{args.hidden}-{args.hidden}-{int(y_train.max()) + 1}",
        "test_accuracy": {name: final_acc[name] for name, _ in configs},
        "learning_rates": dict(configs),
    }
    metrics_path = out.parent / "optimizer_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"metrics saved to {metrics_path}")

    print()
    print(f"{'optimizer':<12}{'lr':<8}{'test accuracy':<14}")
    for name, lr in configs:
        print(f"{name:<12}{lr:<8}{final_acc[name]:<14.4f}")


if __name__ == "__main__":
    main()
