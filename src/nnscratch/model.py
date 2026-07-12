"""Sequential container that chains layers and drives backpropagation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nnscratch.layers import Dense, Layer
from nnscratch.losses import SoftmaxCrossEntropy


class Sequential:
    """A stack of layers applied in order.

    The container owns the forward pass, the reverse-order backward pass, and
    flat views of all trainable parameters and their gradients so optimizers
    can update them without knowing the layer structure.
    """

    def __init__(self, layers: list[Layer]) -> None:
        """Initialize the model.

        Args:
            layers: Layers applied in the given order.
        """
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Run the input through every layer.

        Args:
            x: Input batch of shape (batch, features).
            training: Training-mode flag forwarded to each layer.

        Returns:
            Output of the final layer (logits for a classifier).
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate through all layers in reverse order.

        Args:
            grad: Gradient of the loss with respect to the model output.

        Returns:
            Gradient of the loss with respect to the model input.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[np.ndarray]:
        """All trainable parameter arrays, in a stable order."""
        return [p for layer in self.layers for p in layer.params]

    def gradients(self) -> list[np.ndarray]:
        """Gradient arrays aligned with ``parameters()``."""
        return [g for layer in self.layers for g in layer.grads]

    def l2_penalty(self) -> float:
        """Total L2 penalty contributed by all Dense layers."""
        return sum(layer.l2_penalty() for layer in self.layers if isinstance(layer, Dense))

    def loss(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: SoftmaxCrossEntropy,
        training: bool = False,
    ) -> float:
        """Full regularized loss on a batch.

        Args:
            x: Input batch.
            y: Integer labels.
            loss_fn: Loss object; its state is updated by this call.
            training: Training-mode flag for the forward pass.

        Returns:
            Data loss plus the L2 penalty.
        """
        logits = self.forward(x, training=training)
        return loss_fn.forward(logits, y) + self.l2_penalty()

    def save_weights(self, path: str | Path) -> None:
        """Save all trainable parameters to a compressed ``.npz`` file.

        Only the raw arrays are stored, in ``parameters()`` order. Loading
        requires rebuilding the same architecture first.

        Args:
            path: Destination file. Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"param_{i}": p for i, p in enumerate(self.parameters())}
        np.savez_compressed(path, **arrays)

    def load_weights(self, path: str | Path) -> None:
        """Load parameters saved by :meth:`save_weights` into this model.

        Args:
            path: Source ``.npz`` file.

        Raises:
            ValueError: If the file's arrays do not match this architecture.
        """
        with np.load(Path(path)) as arrays:
            params = self.parameters()
            if len(arrays.files) != len(params):
                raise ValueError(
                    f"Checkpoint holds {len(arrays.files)} arrays but the model "
                    f"has {len(params)} parameter arrays."
                )
            for i, p in enumerate(params):
                saved = arrays[f"param_{i}"]
                if saved.shape != p.shape:
                    raise ValueError(
                        f"Shape mismatch for parameter {i}: "
                        f"checkpoint {saved.shape} vs model {p.shape}."
                    )
                p[...] = saved

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicted class labels for a batch.

        Args:
            x: Input batch.

        Returns:
            Integer labels of shape (batch,).
        """
        return self.forward(x, training=False).argmax(axis=1)
