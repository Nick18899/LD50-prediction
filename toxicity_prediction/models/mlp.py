"""MLP model for toxicity prediction."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron for tabular toxicity prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(features)

