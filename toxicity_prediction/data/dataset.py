"""Dataset classes for toxicity prediction."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset


class ToxicityTabularDataset(Dataset):
    """PyTorch Dataset for tabular toxicity data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class MoleculeGraphDataset(InMemoryDataset):
    """PyTorch Geometric Dataset for molecules from SMILES."""

    COMMON_ATOMS = {6, 7, 8, 9, 15, 16, 17, 35, 53}  # C, N, O, F, P, S, Cl, Br, I

    def __init__(
        self,
        data: pd.DataFrame,
        smiles_col: str = "Canonical SMILES",
        target_col: str = "Toxicity Class Numeric",
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.smiles_list = data[smiles_col].tolist()
        self.labels = data[target_col].values.astype(np.int64)

        unique_labels = np.unique(self.labels)
        if not np.array_equal(unique_labels, np.array([0, 1, 2])):
            raise ValueError(f"Expected classes 0, 1, 2. Got: {unique_labels}")

        self.data_list = self._process_molecules()

    def _get_atom_features(self, atom: rdchem.Atom) -> List[float]:
        """Returns feature vector for a single atom."""
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetTotalValence(),
            int(atom.GetHybridization()),
            int(atom.IsInRing()),
            int(atom.GetIsAromatic()),
            int(atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED),
            atom.GetMass() / 100.0,
        ]

        for atomic_num in self.COMMON_ATOMS:
            features.append(1.0 if atom.GetAtomicNum() == atomic_num else 0.0)

        return features

    def _mol_to_graph(self, smiles: str) -> Data:
        """Converts SMILES to graph Data with atom features."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        mol = Chem.AddHs(mol)

        atom_features = [self._get_atom_features(atom) for atom in mol.GetAtoms()]
        node_features = torch.tensor(atom_features, dtype=torch.float)

        edge_indices = []
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            edge_indices += [[begin_idx, end_idx], [end_idx, begin_idx]]

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return Data(x=node_features, edge_index=edge_index)

    def _process_molecules(self) -> List[Data]:
        """Processes all molecules and adds labels."""
        data_list = []
        for smiles, label in zip(self.smiles_list, self.labels):
            try:
                graph = self._mol_to_graph(smiles)
                graph.y = torch.tensor([label], dtype=torch.long)
                data_list.append(graph)
            except Exception as exc:
                print(f"Error processing SMILES '{smiles}': {exc}")
                continue
        return data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


def load_data(file_path: Path) -> pd.DataFrame:
    """Loads pickle data file."""
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def prepare_tabular_data(
    data: pd.DataFrame,
    target_col: str,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Prepares tabular data for MLP training."""
    features = data.drop(columns=[target_col]).values.astype(np.float32)
    labels = data[target_col].values

    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)

    return features_scaled, labels, scaler

