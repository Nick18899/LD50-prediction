"""GCN model for molecular toxicity prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    """Graph Convolutional Network for molecular classification."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the GCN."""
        hidden = node_features
        for conv, bn in zip(self.convs, self.bns):
            hidden = conv(hidden, edge_index)
            hidden = bn(hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        pooled = global_mean_pool(hidden, batch)
        output = self.classifier(pooled)
        return output

