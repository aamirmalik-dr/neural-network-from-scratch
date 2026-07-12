# Data directory

## Two moons (committed, fully offline)

`two_moons.csv` is a 400 point synthetic 2D binary classification set, two
interleaving half circles with Gaussian noise. It is generated from first
principles by this repository's own code (no scikit-learn generator involved)
with a fixed seed, so it is exactly reproducible and license free:

```
python scripts/make_toy_data.py
```

This file powers the 30 second offline quickstart
(`python scripts/plot_decision_boundary.py`) and part of the test suite.

## Digits (no download needed)

The demos and tests also use the 8x8 handwritten digits dataset that ships
with scikit-learn (1797 samples, 64 features). It loads directly via
`sklearn.datasets.load_digits`, so nothing needs to be placed here.

## MNIST (optional, not committed)

To use the full 28x28 MNIST dataset (70000 samples, 784 features), run:

```
python scripts/download_data.py
```

This fetches `mnist_784` from OpenML through `sklearn.datasets.fetch_openml`
and caches it here as `mnist_784.npz` (roughly 14 MB, gitignored). If the
download fails, the script reports the failure and falls back to the digits
dataset.
