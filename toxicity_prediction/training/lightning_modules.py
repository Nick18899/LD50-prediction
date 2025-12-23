"""PyTorch Lightning modules for training."""

from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import Adam

from toxicity_prediction.models.gcn import GCN
from toxicity_prediction.models.mlp import MLP


class MLPLightningModule(pl.LightningModule):
    """Lightning module for MLP training."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(features)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        self.validation_step_outputs.append({"loss": loss, "probs": probs, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        """Calculate validation metrics at epoch end."""
        outputs = self.validation_step_outputs
        all_probs = torch.cat([out["probs"] for out in outputs]).cpu().numpy()
        all_labels = torch.cat([out["labels"] for out in outputs]).cpu().numpy()
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()

        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        preds = np.argmax(all_probs, axis=1)
        accuracy = accuracy_score(all_labels, preds)
        macro_f1 = f1_score(all_labels, preds, average="macro")

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_roc_auc", roc_auc, prog_bar=True)
        self.log("val_accuracy", accuracy)
        self.log("val_f1", macro_f1)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        features, labels = batch
        outputs = self(features)
        probs = torch.softmax(outputs, dim=1)
        self.test_step_outputs.append({"probs": probs, "labels": labels})

    def on_test_epoch_end(self) -> Dict[str, float]:
        """Calculate test metrics at epoch end."""
        outputs = self.test_step_outputs
        all_probs = torch.cat([out["probs"] for out in outputs]).cpu().numpy()
        all_labels = torch.cat([out["labels"] for out in outputs]).cpu().numpy()

        preds = np.argmax(all_probs, axis=1)
        metrics = {
            "test_accuracy": accuracy_score(all_labels, preds),
            "test_f1": f1_score(all_labels, preds, average="macro"),
            "test_roc_auc": roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"),
        }

        for name, value in metrics.items():
            self.log(name, value)

        self.test_step_outputs.clear()
        return metrics

    def configure_optimizers(self):
        """Configure optimizer."""
        return Adam(self.parameters(), lr=self.learning_rate)


class GCNLightningModule(pl.LightningModule):
    """Lightning module for GCN training."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 256,
        num_classes: int = 3,
        num_layers: int = 8,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GCN(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, node_features, edge_index, batch):
        """Forward pass."""
        return self.model(node_features, edge_index, batch)

    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(outputs, batch.y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(outputs, batch.y)
        probs = torch.softmax(outputs, dim=1)
        self.validation_step_outputs.append({"loss": loss, "probs": probs, "labels": batch.y})
        return loss

    def on_validation_epoch_end(self):
        """Calculate validation metrics at epoch end."""
        outputs = self.validation_step_outputs
        all_probs = torch.cat([out["probs"] for out in outputs]).cpu().numpy()
        all_labels = torch.cat([out["labels"] for out in outputs]).cpu().numpy()
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()

        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        preds = np.argmax(all_probs, axis=1)
        accuracy = accuracy_score(all_labels, preds)
        macro_f1 = f1_score(all_labels, preds, average="macro")

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_roc_auc", roc_auc, prog_bar=True)
        self.log("val_accuracy", accuracy)
        self.log("val_f1", macro_f1)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(outputs, dim=1)
        self.test_step_outputs.append({"probs": probs, "labels": batch.y})

    def on_test_epoch_end(self) -> Dict[str, float]:
        """Calculate test metrics at epoch end."""
        outputs = self.test_step_outputs
        all_probs = torch.cat([out["probs"] for out in outputs]).cpu().numpy()
        all_labels = torch.cat([out["labels"] for out in outputs]).cpu().numpy()

        preds = np.argmax(all_probs, axis=1)
        metrics = {
            "test_accuracy": accuracy_score(all_labels, preds),
            "test_f1": f1_score(all_labels, preds, average="macro"),
            "test_roc_auc": roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"),
        }

        for name, value in metrics.items():
            self.log(name, value)

        self.test_step_outputs.clear()
        return metrics

    def configure_optimizers(self):
        """Configure optimizer."""
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)

