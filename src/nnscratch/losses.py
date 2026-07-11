"""Loss functions."""

from __future__ import annotations

import numpy as np


class SoftmaxCrossEntropy:
    """Softmax activation fused with cross entropy loss.

    Fusing the two gives the numerically stable gradient ``probs - onehot``
    and avoids computing an explicit Jacobian for the softmax.
    """

    def __init__(self) -> None:
        self._probs: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute mean cross entropy over the batch.

        Args:
            logits: Unnormalized scores of shape (batch, classes).
            labels: Integer class labels of shape (batch,).

        Returns:
            Scalar mean cross entropy.
        """
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp
        self._probs = np.exp(log_probs)
        self._labels = labels
        batch = logits.shape[0]
        return float(-log_probs[np.arange(batch), labels].mean())

    def backward(self) -> np.ndarray:
        """Gradient of the mean loss with respect to the logits.

        Returns:
            Array of shape (batch, classes).
        """
        if self._probs is None or self._labels is None:
            raise RuntimeError("backward called before forward")
        batch = self._probs.shape[0]
        grad = self._probs.copy()
        grad[np.arange(batch), self._labels] -= 1.0
        return grad / batch

    def predictions(self) -> np.ndarray:
        """Class predictions from the most recent forward pass."""
        if self._probs is None:
            raise RuntimeError("predictions requested before forward")
        return self._probs.argmax(axis=1)
