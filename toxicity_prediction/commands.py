"""CLI commands for toxicity prediction."""

import pickle
import socket
import subprocess
from pathlib import Path

import fire
import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from toxicity_prediction.data.dataset import (
    MoleculeGraphDataset,
    ToxicityTabularDataset,
    load_data,
    prepare_tabular_data,
)
from toxicity_prediction.data.preprocessing import preprocess_data
from toxicity_prediction.export.onnx_export import export_mlp_to_onnx
from toxicity_prediction.inference.predict import predict_gcn, predict_mlp
from toxicity_prediction.training.lightning_modules import (
    GCNLightningModule,
    MLPLightningModule,
)

CONFIG_PATH = Path(__file__).parent.parent / "configs"


def get_config(config_name: str = "config", overrides: list = None) -> DictConfig:
    """Loads Hydra configuration."""
    if overrides is None:
        overrides = []
    with initialize_config_dir(config_dir=str(CONFIG_PATH.absolute()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def get_git_commit_id() -> str:
    """Returns current git commit ID."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def is_mlflow_server_available(uri: str, timeout: float = 2.0) -> bool:
    """Check if MLflow server is available."""
    try:
        # Parse host and port from URI
        if "://" in uri:
            uri = uri.split("://")[1]
        host, port = uri.split(":")
        port = int(port)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def setup_logger(cfg: DictConfig, log_dir: Path):
    """Setup logger - MLflow if available, otherwise CSV."""
    if is_mlflow_server_available(cfg.mlflow.tracking_uri):
        print(f"MLflow server available at {cfg.mlflow.tracking_uri}")
        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            log_model=True,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        logger.log_hyperparams({"git_commit": get_git_commit_id()})
    else:
        print(f"MLflow server not available at {cfg.mlflow.tracking_uri}")
        print("Using CSV logger instead. To use MLflow, start server with: mlflow server --port 8080")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = CSVLogger(save_dir=str(log_dir), name="training_logs")
    return logger


def preprocess(
    input_file: str = None,
    output_file: str = None,
) -> None:
    """Runs data preprocessing pipeline."""
    cfg = get_config()

    input_path = Path(input_file or cfg.data.raw_file)
    output_path = Path(output_file or cfg.data.processed_file)

    preprocess_data(
        input_path=input_path,
        output_path=output_path,
        const_thresh=cfg.data.preprocessing.const_thresh,
        variance_thresh=cfg.data.preprocessing.variance_thresh,
        corr_thresh=cfg.data.preprocessing.corr_thresh,
    )


def train_mlp(
    data_file: str = None,
    model_name: str = "mlp",
) -> None:
    """Trains MLP model."""
    cfg = get_config(overrides=[f"model={model_name}"])
    pl.seed_everything(cfg.seed)

    data_path = Path(data_file or cfg.data.processed_file)
    data = load_data(data_path)
    data = data.drop(columns=[cfg.data.smiles_col])

    print(f"Loaded data with shape: {data.shape}")

    # Prepare data splits
    features, labels, scaler = prepare_tabular_data(data, cfg.data.target_col)

    indices = np.arange(len(labels))
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=cfg.data.split.test_size,
        stratify=labels,
        random_state=cfg.seed,
    )
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=cfg.data.split.val_size / (1 - cfg.data.split.test_size),
        stratify=labels[idx_train_val],
        random_state=cfg.seed,
    )

    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    train_dataset = ToxicityTabularDataset(features[idx_train], labels[idx_train])
    val_dataset = ToxicityTabularDataset(features[idx_val], labels[idx_val])
    test_dataset = ToxicityTabularDataset(features[idx_test], labels[idx_test])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size)

    # Setup logger
    output_dir = Path(cfg.paths.output_dir)
    logger = setup_logger(cfg, output_dir / "logs")

    # Create model
    input_dim = features.shape[1]
    model = MLPLightningModule(
        input_dim=input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        learning_rate=cfg.training.learning_rate,
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_roc_auc", patience=cfg.training.patience, mode="max"),
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            monitor="val_roc_auc",
            mode="max",
            save_top_k=1,
            filename="mlp-{epoch:02d}-{val_roc_auc:.4f}",
        ),
    ]

    # Train
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Export to ONNX
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "mlp_toxicity.onnx"
    export_mlp_to_onnx(model.model, input_dim, onnx_path)

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    print(f"Model saved to {onnx_path}")
    print(f"Scaler saved to {scaler_path}")


def train_gcn(
    data_file: str = None,
    model_name: str = "gcn",
) -> None:
    """Trains GCN model."""
    cfg = get_config(overrides=[f"model={model_name}"])
    pl.seed_everything(cfg.seed)

    data_path = Path(data_file or cfg.data.processed_file)
    data = load_data(data_path)

    print(f"Loaded data with shape: {data.shape}")
    print("Creating graph dataset (this may take a few minutes)...")

    # Create graph dataset
    dataset = MoleculeGraphDataset(
        data[[cfg.data.smiles_col, cfg.data.target_col]],
        smiles_col=cfg.data.smiles_col,
        target_col=cfg.data.target_col,
    )

    print(f"Dataset created with {len(dataset)} molecules")

    labels = [graph.y.item() for graph in dataset]
    indices = np.arange(len(dataset))

    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=cfg.data.split.test_size,
        stratify=labels,
        random_state=cfg.seed,
    )
    labels_train_val = [labels[idx] for idx in idx_train_val]
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=cfg.data.split.val_size / (1 - cfg.data.split.test_size),
        stratify=labels_train_val,
        random_state=cfg.seed,
    )

    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    train_loader = GeometricDataLoader(
        [dataset[idx] for idx in idx_train],
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = GeometricDataLoader(
        [dataset[idx] for idx in idx_val],
        batch_size=cfg.training.batch_size,
    )
    test_loader = GeometricDataLoader(
        [dataset[idx] for idx in idx_test],
        batch_size=cfg.training.batch_size,
    )

    # Setup logger
    output_dir = Path(cfg.paths.output_dir)
    logger = setup_logger(cfg, output_dir / "logs")

    # Create model
    num_node_features = dataset[0].x.shape[1]
    model = GCNLightningModule(
        num_node_features=num_node_features,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        learning_rate=cfg.training.learning_rate,
    )

    callbacks = [
        EarlyStopping(monitor="val_roc_auc", patience=cfg.training.patience, mode="max"),
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            monitor="val_roc_auc",
            mode="max",
            save_top_k=1,
            filename="gcn-{epoch:02d}-{val_roc_auc:.4f}",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Save PyTorch model
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "gcn_toxicity.pth"
    torch.save(model.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def infer(
    model_path: str,
    data_file: str,
    model_type: str = "mlp",
    scaler_path: str = None,
) -> None:
    """Runs inference with trained model."""
    cfg = get_config()

    data = load_data(Path(data_file))

    if model_type == "mlp":
        if scaler_path is None:
            scaler_path = str(Path(cfg.paths.output_dir) / "scaler.pkl")
        predictions = predict_mlp(
            model_path,
            data,
            scaler_path,
            cfg.data.target_col,
            cfg.data.smiles_col,
        )
    else:
        predictions = predict_gcn(
            model_path, data, cfg.data.smiles_col, cfg.data.target_col
        )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Class distribution: {np.bincount(predictions)}")
    return predictions


def main():
    """Main CLI entry point."""
    fire.Fire(
        {
            "preprocess": preprocess,
            "train-mlp": train_mlp,
            "train-gcn": train_gcn,
            "infer": infer,
        }
    )


if __name__ == "__main__":
    main()
