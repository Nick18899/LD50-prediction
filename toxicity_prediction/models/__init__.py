"""Model definitions for toxicity prediction."""

from toxicity_prediction.models.gcn import GCN
from toxicity_prediction.models.mlp import MLP

__all__ = ["MLP", "GCN"]

