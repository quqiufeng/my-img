#!/usr/bin/env python3
"""
Create Linear 768→2560 projection ONNX model for IPAdapter → Z-Image context.

Initializes with identity for first 768 dims and zero for remaining 1792 dims.
This is functionally equivalent to zero-padding but sets up the infrastructure
for later training. Replace the .data file with trained weights when available.

Usage:
  python3 scripts/create_ipadapter_proj.py

Output:
  /data/models/image/ipadapter_proj.onnx
  /data/models/image/ipadapter_proj.onnx.data
"""
import os, sys
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, save
except ImportError:
    print("[ERROR] onnx not installed. Run: pip install onnx")
    sys.exit(1)

OUTPUT_DIR = "/data/models/image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 768
OUTPUT_DIM = 2560

print(f"[INFO] Creating Linear({INPUT_DIM}→{OUTPUT_DIM}) projection model")
print(f"[INFO] Output: {OUTPUT_DIR}/ipadapter_proj.onnx")

# --- Weights: identity-like matrix ---
# ONNX MatMul: output = input @ W^T + B
#   input shape: [1, INPUT_DIM]  = [1, 768]
#   W shape:     [OUTPUT_DIM, INPUT_DIM] = [2560, 768]  → MatMul treats this as W, not W^T!
# Actually ONNX MatMul(A, B): A @ B, inner dims must match.
#   A=[1, 768], B must be [768, OUTPUT_DIM] = [768, 2560]
#   output = [1, 768] @ [768, 2560] = [1, 2560]
# So W shape is [INPUT_DIM, OUTPUT_DIM] = [768, 2560]
W = np.zeros((INPUT_DIM, OUTPUT_DIM), dtype=np.float32)
for i in range(INPUT_DIM):
    W[i, i] = 1.0  # identity diagonal: input[i] → output[i]

# Bias: all zeros
b = np.zeros((OUTPUT_DIM,), dtype=np.float32)

print(f"  Weight shape: {W.shape} (768×2560)")
print(f"  Weight stats: min={W.min():.4f}, max={W.max():.4f}, nonzero={np.count_nonzero(W)}/{W.size}")
print(f"  Bias stats:   min={b.min():.4f}, max={b.max():.4f}")

# --- Build ONNX graph ---
X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, INPUT_DIM])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, OUTPUT_DIM])

W_tensor = helper.make_tensor('W', TensorProto.FLOAT, [INPUT_DIM, OUTPUT_DIM], W.tobytes(), raw=True)
b_tensor = helper.make_tensor('B', TensorProto.FLOAT, [OUTPUT_DIM], b.tobytes(), raw=True)

matmul_node = helper.make_node(
    'MatMul', ['input', 'W'], ['matmul_out'], name='MatMul')
add_node = helper.make_node(
    'Add', ['matmul_out', 'B'], ['output'], name='Add')

graph = helper.make_graph(
    [matmul_node, add_node],
    'ipadapter_proj',
    [X],
    [Y],
    [W_tensor, b_tensor])

opset = helper.make_opsetid('', 14)
model = helper.make_model(graph, opset_imports=[opset], ir_version=7)

# Use external data format for the weight file
output_path = os.path.join(OUTPUT_DIR, "ipadapter_proj.onnx")
onnx.save_model(
    model,
    output_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location='ipadapter_proj.onnx.data',
    size_threshold=0)

data_path = os.path.join(OUTPUT_DIR, "ipadapter_proj.onnx.data")
onnx_size = os.path.getsize(output_path) / 1024
data_size = os.path.getsize(data_path) / (1024 * 1024)
print(f"\n[OK] Model saved:")
print(f"  ONNX:       {output_path} ({onnx_size:.1f} KB)")
print(f"  Data:       {data_path} ({data_size:.2f} MB)")
print(f"  Parameters: {W.size:,} weights + {b.size:,} biases")
print(f"\n[INFO] Verification - test inference:")
print(f"  Input:  random [1, {INPUT_DIM}]")
print(f"  Output: should be [1, {OUTPUT_DIM}] with first {INPUT_DIM} dims ≈ input, rest ≈ 0")

# Quick verification
import onnxruntime as ort
session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
test_in = np.random.randn(1, INPUT_DIM).astype(np.float32)
test_out = session.run(['output'], {'input': test_in})[0]
print(f"  Shape:  {test_out.shape}")
print(f"  Max err in first {INPUT_DIM} dims: {np.max(np.abs(test_out[0, :INPUT_DIM] - test_in[0])):.6f}")
print(f"  Max err in last {OUTPUT_DIM - INPUT_DIM} dims: {np.max(np.abs(test_out[0, INPUT_DIM:])):.6f}")
