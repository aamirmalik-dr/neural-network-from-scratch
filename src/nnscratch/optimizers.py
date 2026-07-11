"""First-order optimizers: SGD, SGD with momentum, and Adam.

Optimizers update parameter arrays in place. Per-parameter state (velocity,
moment estimates) is created lazily on the first ``step`` call and keyed by
position, so the same list of parameters must be passed on every step.
"""

from __future__ import annotations

import numpy as np


class Optimizer:
    """Base class for optimizers."""

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Apply one update to every parameter array in place.

        Args:
            params: Parameter arrays to update.
            grads: Gradient arrays aligned with ``params``.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Vanilla stochastic gradient descent."""

    def __init__(self, lr: float = 0.01) -> None:
        """Initialize the optimizer.

        Args:
            lr: Learning rate.
        """
        self.lr = float(lr)

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        for p, g in zip(params, grads, strict=True):
            p -= self.lr * g


class Momentum(Optimizer):
    """SGD with classical momentum.

    Maintains a velocity ``v = mu * v - lr * g`` per parameter and applies
    ``p += v``.
    """

    def __init__(self, lr: float = 0.01, mu: float = 0.9) -> None:
        """Initialize the optimizer.

        Args:
            lr: Learning rate.
            mu: Momentum coefficient in [0, 1).
        """
        self.lr = float(lr)
        self.mu = float(mu)
        self._velocity: list[np.ndarray] | None = None

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self._velocity is None:
            self._velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self._velocity, strict=True):
            v *= self.mu
            v -= self.lr * g
            p += v


class Adam(Optimizer):
    """Adam optimizer with bias-corrected first and second moment estimates."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the optimizer.

        Args:
            lr: Learning rate.
            beta1: Exponential decay rate for the first moment.
            beta2: Exponential decay rate for the second moment.
            eps: Small constant added to the denominator for stability.
        """
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None
        self._t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self._m is None or self._v is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self._t += 1
        bc1 = 1.0 - self.beta1**self._t
        bc2 = 1.0 - self.beta2**self._t
        for p, g, m, v in zip(params, grads, self._m, self._v, strict=True):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "adam": Adam,
}


def get_optimizer(name: str, lr: float) -> Optimizer:
    """Create an optimizer by name with the given learning rate.

    Args:
        name: One of "sgd", "momentum", or "adam".
        lr: Learning rate.

    Returns:
        A fresh optimizer instance.

    Raises:
        ValueError: If the name is not a known optimizer.
    """
    try:
        cls = OPTIMIZERS[name]
    except KeyError as err:
        known = ", ".join(sorted(OPTIMIZERS))
        raise ValueError(f"Unknown optimizer '{name}'. Known: {known}") from err
    return cls(lr=lr)
