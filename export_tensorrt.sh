#!/bin/bash
# Export ONNX model to TensorRT
# Usage: ./export_tensorrt.sh <onnx_model_path> <trt_model_path>

set -e

ONNX_MODEL=${1:-"outputs/mlp_toxicity.onnx"}
TRT_MODEL=${2:-"outputs/mlp_toxicity.trt"}

echo "Converting ONNX model to TensorRT..."
echo "Input: $ONNX_MODEL"
echo "Output: $TRT_MODEL"

trtexec --onnx="$ONNX_MODEL" --saveEngine="$TRT_MODEL" --fp16

echo "Conversion complete!"

