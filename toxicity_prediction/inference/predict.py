"""Inference utilities."""

import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from toxicity_prediction.data.dataset import MoleculeGraphDataset
from toxicity_prediction.models.gcn import GCN


def predict_mlp(
    model_path: str,
    data: pd.DataFrame,
    scaler_path: str,
    target_col: str = "Toxicity Class Numeric",
    smiles_col: str = "Canonical SMILES",
) -> np.ndarray:
    """Runs MLP inference using ONNX model."""
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    # Remove non-feature columns
    columns_to_drop = []
    if target_col in data.columns:
        columns_to_drop.append(target_col)
    if smiles_col in data.columns:
        columns_to_drop.append(smiles_col)

    if columns_to_drop:
        features = data.drop(columns=columns_to_drop).values.astype(np.float32)
    else:
        features = data.values.astype(np.float32)

    features_scaled = scaler.transform(features)

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features_scaled})

    predictions = np.argmax(outputs[0], axis=1)
    return predictions


def predict_gcn(
    model_path: str,
    data: pd.DataFrame,
    smiles_col: str = "Canonical SMILES",
    target_col: str = "Toxicity Class Numeric",
    num_node_features: int = 19,
    hidden_dim: int = 256,
    num_layers: int = 8,
) -> np.ndarray:
    """Runs GCN inference."""
    model = GCN(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        num_classes=3,
        num_layers=num_layers,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dataset = MoleculeGraphDataset(
        data[[smiles_col, target_col]],
        smiles_col=smiles_col,
        target_col=target_col,
    )

    predictions = []
    with torch.no_grad():
        for graph in dataset:
            batch = torch.zeros(graph.x.size(0), dtype=torch.long)
            output = model(graph.x, graph.edge_index, batch)
            pred = torch.argmax(output, dim=1).item()
            predictions.append(pred)

    return np.array(predictions)

