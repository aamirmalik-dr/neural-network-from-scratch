"""Dataset loading utilities.

Three datasets are supported:

* ``two moons``: a 2D synthetic binary classification set generated here from
  first principles with NumPy (two interleaving half circles plus Gaussian
  noise). A fixed-seed copy is committed at ``data/two_moons.csv`` so the
  quickstart runs with no network at all.
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
TWO_MOONS_CSV = "two_moons.csv"


def make_two_moons(
    n_samples: int = 400,
    noise: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the two interleaving half moons dataset with plain NumPy.

    The upper moon is the top half of the unit circle. The lower moon is the
    bottom half of a unit circle shifted right by 1 and up by 0.5, so the two
    arcs interleave and no straight line separates them. Isotropic Gaussian
    noise is added to every point.

    Args:
        n_samples: Total number of points, split evenly between the classes.
        noise: Standard deviation of the Gaussian noise added to each point.
        rng: Random generator. A fresh default generator is used when omitted.

    Returns:
        Tuple ``(x, y)`` where ``x`` has shape (n_samples, 2) and ``y`` holds
        integer labels in {0, 1}.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_upper = n_samples // 2
    n_lower = n_samples - n_upper
    theta_upper = rng.uniform(0.0, np.pi, size=n_upper)
    theta_lower = rng.uniform(0.0, np.pi, size=n_lower)
    upper = np.column_stack([np.cos(theta_upper), np.sin(theta_upper)])
    lower = np.column_stack([1.0 - np.cos(theta_lower), 0.5 - np.sin(theta_lower)])
    x = np.vstack([upper, lower]) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.concatenate([np.zeros(n_upper, dtype=np.int64), np.ones(n_lower, dtype=np.int64)])
    order = rng.permutation(n_samples)
    return x[order], y[order]


def load_two_moons(
    path: str | Path = Path("data") / TWO_MOONS_CSV, test_size: float = 0.25, seed: int = 0
) -> Dataset:
    """Load the committed two moons CSV and split it for training.

    Args:
        path: CSV file with columns x1, x2, label (header row included).
        test_size: Fraction of the data held out for testing.
        seed: Random seed for the stratified split.

    Returns:
        Tuple (x_train, x_test, y_train, y_test).

    Raises:
        FileNotFoundError: If the CSV is missing. Regenerate it with
            ``python scripts/make_toy_data.py``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate it with: python scripts/make_toy_data.py"
        )
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    x = raw[:, :2]
    y = raw[:, 2].astype(np.int64)
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


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
        name: "moons", "digits", or "mnist".
        data_dir: Directory holding the two moons CSV and the MNIST cache.
        seed: Random seed for the split.

    Returns:
        Tuple (x_train, x_test, y_train, y_test).

    Raises:
        ValueError: If the name is not a known dataset.
    """
    if name == "moons":
        return load_two_moons(path=Path(data_dir) / TWO_MOONS_CSV, seed=seed)
    if name == "digits":
        return load_digits_data(seed=seed)
    if name == "mnist":
        return load_mnist_data(data_dir=data_dir, seed=seed)
    raise ValueError(f"Unknown dataset '{name}'. Known: moons, digits, mnist")
