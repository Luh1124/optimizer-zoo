"""Simple MLP for MNIST classification.

3-layer MLP: 784 -> 256 -> 128 -> 10
Fastest benchmark (~1 min on GPU). Use this for quick sanity checks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple 3-layer MLP for MNIST (28x28 grayscale -> 10 classes).

    Args:
        input_dim: Flattened input dimension (default: 784 = 28*28).
        hidden_dims: Tuple of hidden layer sizes.
        num_classes: Number of output classes.
        dropout: Dropout rate between layers.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784).

        Returns:
            Logits of shape (batch, num_classes).
        """
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)
