"""Download MNIST from OpenML into the local data directory.

Falls back to the bundled scikit-learn digits dataset if the download fails,
and prints which dataset ended up available.

Usage:
    python scripts/download_data.py [--data-dir data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nnscratch.data import load_digits_data, load_mnist_data


def main() -> None:
    """Fetch MNIST with a graceful fallback to the digits dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory for the MNIST cache")
    args = parser.parse_args()

    try:
        x_train, x_test, _, _ = load_mnist_data(data_dir=args.data_dir)
        cache = Path(args.data_dir) / "mnist_784.npz"
        print(f"Using dataset: mnist (cached at {cache})")
        print(f"Train samples: {x_train.shape[0]}, test samples: {x_test.shape[0]}")
    except RuntimeError as err:
        print(f"MNIST unavailable ({err}), falling back to the digits dataset.")
        x_train, x_test, _, _ = load_digits_data()
        print("Using dataset: digits (bundled with scikit-learn, no download needed)")
        print(f"Train samples: {x_train.shape[0]}, test samples: {x_test.shape[0]}")


if __name__ == "__main__":
    main()
