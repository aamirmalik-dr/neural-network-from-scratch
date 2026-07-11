"""Weight initialization schemes for dense layers."""

from __future__ import annotations

import numpy as np


def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Glorot (Xavier) normal initialization.

    Draws weights from N(0, 2 / (fan_in + fan_out)). Suited to layers followed
    by symmetric activations such as sigmoid or tanh.

    Args:
        fan_in: Number of input units.
        fan_out: Number of output units.
        rng: NumPy random generator used to draw the weights.

    Returns:
        Weight matrix of shape (fan_in, fan_out).
    """
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.standard_normal((fan_in, fan_out)) * std


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He (Kaiming) normal initialization.

    Draws weights from N(0, 2 / fan_in). Suited to layers followed by ReLU.

    Args:
        fan_in: Number of input units.
        fan_out: Number of output units.
        rng: NumPy random generator used to draw the weights.

    Returns:
        Weight matrix of shape (fan_in, fan_out).
    """
    std = np.sqrt(2.0 / fan_in)
    return rng.standard_normal((fan_in, fan_out)) * std


INITIALIZERS = {
    "xavier": xavier_init,
    "he": he_init,
}


def get_initializer(name: str):
    """Look up an initializer by name.

    Args:
        name: One of "xavier" or "he".

    Returns:
        The initializer function.

    Raises:
        ValueError: If the name is not a known initializer.
    """
    try:
        return INITIALIZERS[name]
    except KeyError as err:
        known = ", ".join(sorted(INITIALIZERS))
        raise ValueError(f"Unknown initializer '{name}'. Known: {known}") from err
