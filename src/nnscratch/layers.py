"""Layers for a feedforward network: Dense, ReLU, Sigmoid, and inverted Dropout.

Every layer exposes ``forward`` and ``backward``. Trainable layers additionally
expose ``params`` and ``grads`` lists whose entries stay index-aligned, so an
optimizer can update them in lockstep.
"""

from __future__ import annotations

import numpy as np

from nnscratch.initializers import get_initializer


class Layer:
    """Base class for all layers."""

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Compute the layer output for a batch of inputs.

        Args:
            x: Input batch of shape (batch, features).
            training: Whether the network is in training mode. Affects layers
                such as Dropout that behave differently at inference time.

        Returns:
            Output batch.
        """
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backpropagate the upstream gradient through the layer.

        Args:
            grad_out: Gradient of the loss with respect to this layer's output.

        Returns:
            Gradient of the loss with respect to this layer's input.
        """
        raise NotImplementedError

    @property
    def params(self) -> list[np.ndarray]:
        """Trainable parameter arrays of this layer (may be empty)."""
        return []

    @property
    def grads(self) -> list[np.ndarray]:
        """Gradient arrays aligned with ``params`` (may be empty)."""
        return []


class Dense(Layer):
    """Fully connected layer computing ``y = x @ w + b``.

    Supports optional L2 weight decay. The decay term is added to the weight
    gradient during ``backward``; the matching penalty ``0.5 * l2 * ||w||^2``
    is reported through ``l2_penalty`` so that losses and numerical gradient
    checks stay consistent with the analytic gradients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "he",
        l2: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize the layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            init: Initialization scheme name, "he" or "xavier".
            l2: L2 regularization strength applied to the weights (not bias).
            rng: Random generator for weight initialization. A fresh default
                generator is used when omitted.
        """
        if rng is None:
            rng = np.random.default_rng()
        init_fn = get_initializer(init)
        self.w: np.ndarray = init_fn(in_features, out_features, rng)
        self.b: np.ndarray = np.zeros(out_features)
        self.l2 = float(l2)
        self.grad_w: np.ndarray = np.zeros_like(self.w)
        self.grad_b: np.ndarray = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._x = x
        return x @ self.w + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("backward called before forward")
        self.grad_w = self._x.T @ grad_out + self.l2 * self.w
        self.grad_b = grad_out.sum(axis=0)
        return grad_out @ self.w.T

    def l2_penalty(self) -> float:
        """Return the L2 penalty term contributed by this layer."""
        if self.l2 == 0.0:
            return 0.0
        return 0.5 * self.l2 * float(np.sum(self.w * self.w))

    @property
    def params(self) -> list[np.ndarray]:
        return [self.w, self.b]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.grad_w, self.grad_b]


class ReLU(Layer):
    """Rectified linear activation, ``max(0, x)`` elementwise."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._mask = x > 0
        return np.where(self._mask, x, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("backward called before forward")
        return grad_out * self._mask


class Sigmoid(Layer):
    """Logistic sigmoid activation, ``1 / (1 + exp(-x))`` elementwise."""

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        # Numerically stable piecewise evaluation.
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        self._out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._out is None:
            raise RuntimeError("backward called before forward")
        return grad_out * self._out * (1.0 - self._out)


class Dropout(Layer):
    """Inverted dropout.

    During training each unit is zeroed with probability ``rate`` and the
    surviving units are scaled by ``1 / (1 - rate)``, so no rescaling is
    needed at inference time. At inference the layer is the identity.
    """

    def __init__(self, rate: float, rng: np.random.Generator | None = None) -> None:
        """Initialize the layer.

        Args:
            rate: Probability of dropping a unit, in [0, 1).
            rng: Random generator for the dropout masks.
        """
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"rate must be in [0, 1), got {rate}")
        self.rate = float(rate)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if not training or self.rate == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.rate
        self._mask = (self.rng.random(x.shape) < keep) / keep
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad_out
        return grad_out * self._mask
