# Data directory

Datasets are not committed to this repository. This directory only holds local caches.

## Digits (default, no download needed)

The demos and tests use the 8x8 handwritten digits dataset that ships with scikit-learn (1797 samples, 64 features). It loads directly via `sklearn.datasets.load_digits`, so nothing needs to be placed here.

## MNIST (optional)

To use the full 28x28 MNIST dataset (70000 samples, 784 features), run:

```
python scripts/download_data.py
```

This fetches `mnist_784` from OpenML through `sklearn.datasets.fetch_openml` and caches it here as `mnist_784.npz` (roughly 16 MB, gitignored). If the download fails, the script reports the failure and falls back to the digits dataset.
