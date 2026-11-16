"""Compact multi-class PyTorch model for sensor-feature classification."""

from __future__ import annotations

import torch
from torch import nn


class VibrationMLP(nn.Module):
    """Small MLP chosen for CPU-friendly inference and edge deployment experiments."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, int] = (96, 48),
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes < 2:
            raise ValueError("input_dim must be positive and num_classes must be at least two.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if len(hidden_dims) != 2 or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain exactly two positive layer widths.")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.dropout_rate = float(dropout)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits (CrossEntropyLoss expects logits)."""
        return self.network(inputs)
