"""Dataset loading utilities.

Two datasets are supported:

* ``digits``: the 8x8 handwritten digits set bundled with scikit-learn
  (1797 samples, 64 features). Fully offline, used by the tests and demos.
* ``mnist``: the full 28x28 MNIST set (70000 samples, 784 features) fetched
  from OpenML on first use and cached as a compressed .npz in the data
  directory. Falls back to ``digits`` if the download fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split

Dataset = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

MNIST_CACHE = "mnist_784.npz"


def load_digits_data(test_size: float = 0.2, seed: int = 0) -> Dataset:
    """Load the scikit-learn digits dataset, scaled to [0, 1] and split.

    Args:
        test_size: Fraction of the data held out for testing.
        seed: Random seed for the stratified split.

    Returns:
        Tuple (x_train, x_test, y_train, y_test).
    """
    digits = load_digits()
    x = digits.data / 16.0
    y = digits.target.astype(np.int64)
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


def load_mnist_data(
    data_dir: str | Path = "data", test_size: float = 1 / 7, seed: int = 0
) -> Dataset:
    """Load MNIST, downloading from OpenML and caching locally on first use.

    Args:
        data_dir: Directory used to cache the downloaded arrays.
        test_size: Fraction of the data held out for testing. The default
            reproduces the conventional 60000/10000 split size.
        seed: Random seed for the stratified split.

    Returns:
        Tuple (x_train, x_test, y_train, y_test) with pixels scaled to [0, 1].

    Raises:
        RuntimeError: If the cache is missing and the download fails.
    """
    data_dir = Path(data_dir)
    cache = data_dir / MNIST_CACHE
    if cache.exists():
        arrays = np.load(cache)
        x, y = arrays["x"], arrays["y"]
    else:
        try:
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        except Exception as err:
            raise RuntimeError(f"MNIST download failed: {err}") from err
        x = mnist.data.astype(np.uint8)
        y = mnist.target.astype(np.int64)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=x, y=y)
    x = x.astype(np.float64) / 255.0
    y = y.astype(np.int64)
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


def load_dataset(name: str, data_dir: str | Path = "data", seed: int = 0) -> Dataset:
    """Load a dataset by name.

    Args:
        name: "digits" or "mnist".
        data_dir: Cache directory used for MNIST.
        seed: Random seed for the split.

    Returns:
        Tuple (x_train, x_test, y_train, y_test).

    Raises:
        ValueError: If the name is not a known dataset.
    """
    if name == "digits":
        return load_digits_data(seed=seed)
    if name == "mnist":
        return load_mnist_data(data_dir=data_dir, seed=seed)
    raise ValueError(f"Unknown dataset '{name}'. Known: digits, mnist")
