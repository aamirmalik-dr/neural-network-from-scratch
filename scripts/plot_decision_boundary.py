"""Offline quickstart: train on the committed two moons set and plot the boundary.

Loads data/two_moons.csv (committed, no network), runs a numerical gradient
check on the exact architecture about to be trained, trains it with Adam,
and writes:

* results/decision_boundary.png, the decision boundary with a loss inset
* results/metrics.json, the measured numbers backing the README
* results/two_moons_model.npz, the trained weights for instant inference

Usage:
    python scripts/plot_decision_boundary.py [--epochs 200] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from nnscratch.data import load_two_moons
from nnscratch.gradient_check import gradient_check
from nnscratch.layers import Dense, ReLU
from nnscratch.losses import SoftmaxCrossEntropy
from nnscratch.model import Sequential
from nnscratch.optimizers import Adam
from nnscratch.training import accuracy, train

# Categorical slots and chrome from the repo's figure palette.
CLASS_COLORS = ("#2a78d6", "#e34948")
SURFACE = "#fcfcfb"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Diverging ramp for P(class 1): blue, neutral gray midpoint, red.
PROB_CMAP = LinearSegmentedColormap.from_list("moons_diverging", ["#2a78d6", "#f0efec", "#e34948"])


def build_model(seed: int, hidden: int = 32) -> Sequential:
    """Build the small two moons classifier.

    Args:
        seed: Seed for weight initialization.
        hidden: Hidden layer width.

    Returns:
        A 2-hidden-layer ReLU network with two output logits.
    """
    rng = np.random.default_rng(seed)
    return Sequential(
        [
            Dense(2, hidden, init="he", rng=rng),
            ReLU(),
            Dense(hidden, hidden, init="he", rng=rng),
            ReLU(),
            Dense(hidden, 2, init="xavier", rng=rng),
        ]
    )


def probability_grid(
    model: Sequential, x: np.ndarray, margin: float = 0.6, steps: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate P(class 1) on a dense grid covering the data.

    Args:
        model: Trained classifier with two output logits.
        x: Data points used to set the grid extent.
        margin: Padding added around the data's bounding box.
        steps: Grid resolution per axis.

    Returns:
        Tuple (xx, yy, probs) of meshgrid coordinates and probabilities.
    """
    x_min, x_max = x[:, 0].min() - margin, x[:, 0].max() + margin
    y_min, y_max = x[:, 1].min() - margin, x[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    logits = model.forward(grid, training=False)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = (exp[:, 1] / exp.sum(axis=1)).reshape(xx.shape)
    return xx, yy, probs


def main() -> None:
    """Train on two moons and save the hero figure, metrics, and weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/two_moons.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = load_two_moons(path=args.data, seed=args.seed)
    print(f"two moons: train={x_train.shape} test={x_test.shape}")

    # Correctness proof first: numerical gradient check on this architecture.
    # A few warmup steps move the parameters to a generic point. At exact zero
    # initialization of the biases, a sample whose first hidden layer is fully
    # dead lands the next preactivation exactly on the ReLU kink, where central
    # differences are undefined (see docs/math.md, section 8).
    check_model = build_model(args.seed, hidden=8)
    check_x, check_y = x_train[:16], y_train[:16]
    train(
        check_model,
        Adam(lr=0.01),
        check_x,
        check_y,
        epochs=3,
        batch_size=16,
        rng=np.random.default_rng(args.seed),
    )
    rel_error = gradient_check(check_model, SoftmaxCrossEntropy(), check_x, check_y)
    print(f"gradient check max relative error: {rel_error:.2e}")

    model = build_model(args.seed)
    history = train(
        model,
        Adam(lr=0.01),
        x_train,
        y_train,
        x_val=x_test,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=32,
        rng=np.random.default_rng(args.seed),
    )
    train_acc = accuracy(model, x_train, y_train)
    test_acc = accuracy(model, x_test, y_test)
    print(f"train accuracy: {train_acc:.4f}  test accuracy: {test_acc:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hero figure: diverging probability field, boundary contour, data points.
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xx, yy, probs = probability_grid(model, np.vstack([x_train, x_test]))
    ax.contourf(xx, yy, probs, levels=np.linspace(0.0, 1.0, 21), cmap=PROB_CMAP, alpha=0.55)
    ax.contour(xx, yy, probs, levels=[0.5], colors="#0b0b0b", linewidths=1.4)
    for cls, color, marker in ((0, CLASS_COLORS[0], "o"), (1, CLASS_COLORS[1], "^")):
        mask = y_train == cls
        ax.scatter(
            x_train[mask, 0],
            x_train[mask, 1],
            s=22,
            c=color,
            marker=marker,
            edgecolors=SURFACE,
            linewidths=0.6,
            label=f"class {cls} (train)",
        )
    ax.set_xlabel("x1", color=INK_MUTED)
    ax.set_ylabel("x2", color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_title(
        f"Decision boundary on two moons, test accuracy {test_acc:.3f}",
        color="#0b0b0b",
    )
    ax.legend(loc="upper right", framealpha=0.9)

    inset = ax.inset_axes([0.035, 0.06, 0.34, 0.26])
    inset.plot(range(1, len(history.loss) + 1), history.loss, color="#2a78d6", linewidth=1.6)
    inset.set_yscale("log")
    inset.text(
        0.96,
        0.86,
        "training loss",
        transform=inset.transAxes,
        fontsize=8,
        color=INK_MUTED,
        ha="right",
    )
    inset.tick_params(labelsize=7, colors=INK_MUTED)
    inset.set_facecolor(SURFACE)
    inset.grid(True, color=GRID, linewidth=0.5)
    for spine in inset.spines.values():
        spine.set_color(GRID)

    fig.tight_layout()
    fig_path = out_dir / "decision_boundary.png"
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    print(f"figure saved to {fig_path}")

    weights_path = out_dir / "two_moons_model.npz"
    model.save_weights(weights_path)
    print(f"weights saved to {weights_path}")

    metrics = {
        "dataset": "two_moons (synthetic, committed at data/two_moons.csv, seed 0)",
        "architecture": "2-32-32-2, ReLU, He init, Adam lr 0.01",
        "epochs": args.epochs,
        "seed": args.seed,
        "gradient_check_max_rel_error": rel_error,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "final_training_loss": history.loss[-1],
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
