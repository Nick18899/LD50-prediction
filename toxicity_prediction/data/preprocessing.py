"""Data preprocessing utilities."""

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def discretize_toxicity(ld50_value: float) -> int:
    """
    Discretizes toxicity values into 3 classes.

    Class 0 (High toxicity): LD50 < 50 mg/kg
    Class 1 (Moderate toxicity): LD50 = 50-500 mg/kg
    Class 2 (Low toxicity): LD50 > 500 mg/kg
    """
    if pd.isna(ld50_value):
        return None
    elif ld50_value < 50:
        return 0
    elif ld50_value <= 500:
        return 1
    else:
        return 2


def remove_near_constant_features(
    dataframe: pd.DataFrame, threshold: float = 0.9
) -> List[str]:
    """Removes features where one value appears more than threshold proportion."""
    keep_features = []

    for col in dataframe.columns:
        val_counts = dataframe[col].value_counts(normalize=True)
        max_prop = val_counts.iloc[0]

        if max_prop < threshold:
            keep_features.append(col)

    return keep_features


def remove_low_variance_features(
    dataframe: pd.DataFrame, threshold: float = 0.01
) -> List[str]:
    """Removes features with variance below threshold."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(dataframe)
    return dataframe.columns[selector.get_support()].tolist()


def remove_highly_correlated_features(
    dataframe: pd.DataFrame, threshold: float = 0.85
) -> List[str]:
    """Removes highly correlated features using Spearman correlation."""
    corr_matrix = dataframe.corr("spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [col for col in dataframe.columns if col not in to_drop]


def preprocess_data(
    input_path: Path,
    output_path: Path,
    const_thresh: float = 0.9,
    variance_thresh: float = 0.01,
    corr_thresh: float = 0.85,
) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    with open(input_path, "rb") as file:
        data = pickle.load(file)

    data["Toxicity Value"] = data["Toxicity Value"].apply(lambda val: float(val))
    data = data.drop(columns=["InChIKey", "Source File"])
    data["Toxicity Class Numeric"] = data["Toxicity Value"].apply(discretize_toxicity)
    data = data.drop(columns=["Toxicity Value"])

    protected_cols = ["Canonical SMILES", "Toxicity Class Numeric"]
    feature_cols = [col for col in data.columns if col not in protected_cols]
    data = data.dropna(subset=feature_cols)

    feature_df = data[feature_cols]

    # Step 1: Remove near-constant features
    selected_features = remove_near_constant_features(feature_df, const_thresh)
    feature_df = feature_df[selected_features]
    print(f"Features after constant filter: {len(selected_features)}")

    # Step 2: Remove low variance features
    selected_features = remove_low_variance_features(feature_df, variance_thresh)
    feature_df = feature_df[selected_features]
    print(f"Features after variance filter: {len(selected_features)}")

    # Step 3: Remove highly correlated features
    selected_features = remove_highly_correlated_features(feature_df, corr_thresh)
    feature_df = feature_df[selected_features]
    print(f"Features after correlation filter: {len(selected_features)}")

    result_df = data[selected_features + protected_cols]

    with open(output_path, "wb") as file:
        pickle.dump(result_df, file)

    print(f"Preprocessed data saved to {output_path}")
    return result_df

