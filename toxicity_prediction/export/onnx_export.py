"""ONNX export utilities."""

from pathlib import Path

import torch

from toxicity_prediction.models.mlp import MLP


def export_mlp_to_onnx(
    model: MLP,
    input_dim: int,
    output_path: Path,
    opset_version: int = 18,
) -> None:
    """Exports MLP model to ONNX format."""
    model.eval()

    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["output"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Model exported to ONNX: {output_path}")

