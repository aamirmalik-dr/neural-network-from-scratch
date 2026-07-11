"""Numerical gradient checking via central differences.

Compares analytic backpropagation gradients against numerical estimates of
``d loss / d parameter`` computed with central differences. The comparison
uses the standard symmetric relative error

    ||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric|| + tiny)

per parameter array. With float64 inputs and a well behaved loss the error
should sit far below 1e-5.
"""

from __future__ import annotations

import numpy as np

from nnscratch.losses import SoftmaxCrossEntropy
from nnscratch.model import Sequential


def numerical_gradient(
    model: Sequential,
    loss_fn: SoftmaxCrossEntropy,
    x: np.ndarray,
    y: np.ndarray,
    param: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Estimate the gradient of the loss with respect to one parameter array.

    Args:
        model: Model whose loss is evaluated.
        loss_fn: Loss object.
        x: Input batch.
        y: Integer labels.
        param: Parameter array to perturb (a view into the model, modified in
            place and restored).
        eps: Perturbation size for the central difference.

    Returns:
        Numerical gradient with the same shape as ``param``.
    """
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + eps
        loss_plus = model.loss(x, y, loss_fn, training=False)
        param[idx] = original - eps
        loss_minus = model.loss(x, y, loss_fn, training=False)
        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()
    return grad


def gradient_check(
    model: Sequential,
    loss_fn: SoftmaxCrossEntropy,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compare backpropagation gradients to numerical gradients.

    Args:
        model: Model to check. Should be small; the numerical pass evaluates
            the full loss twice per scalar parameter.
        loss_fn: Loss object.
        x: Input batch.
        y: Integer labels.
        eps: Perturbation size for the central differences.

    Returns:
        The maximum symmetric relative error over all parameter arrays.
    """
    loss_value = model.loss(x, y, loss_fn, training=False)
    del loss_value
    model.backward(loss_fn.backward())
    analytic = [g.copy() for g in model.gradients()]

    max_rel_error = 0.0
    for param, grad_a in zip(model.parameters(), analytic, strict=True):
        grad_n = numerical_gradient(model, loss_fn, x, y, param, eps=eps)
        num = float(np.linalg.norm(grad_a - grad_n))
        den = float(np.linalg.norm(grad_a) + np.linalg.norm(grad_n)) + 1e-12
        max_rel_error = max(max_rel_error, num / den)
    return max_rel_error
