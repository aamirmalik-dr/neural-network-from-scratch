"""Regenerate the committed two moons toy dataset.

The dataset committed at data/two_moons.csv was produced by this script with
its default arguments (seed 0), so it is fully reproducible and license free.

Usage:
    python scripts/make_toy_data.py [--n-samples 400] [--noise 0.2] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nnscratch.data import make_two_moons


def main() -> None:
    """Generate the two moons CSV with a fixed seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/two_moons.csv")
    args = parser.parse_args()

    x, y = make_two_moons(args.n_samples, noise=args.noise, rng=np.random.default_rng(args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([x, y.astype(float)])
    np.savetxt(out, rows, delimiter=",", header="x1,x2,label", comments="", fmt="%.6f")
    print(f"wrote {x.shape[0]} points to {out}")


if __name__ == "__main__":
    main()
