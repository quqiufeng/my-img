#!/usr/bin/env python3
"""
Regenerate clip_vision_vit_h_hidden.onnx from clip_vision_sd15.safetensors.
Outputs hidden_states [1, 257, 1280] with combined external data file.
Run: /data/venv/bin/python3 regenerate_clip_vision_onnx.py
"""
import os, sys
import torch
import torch.nn as nn
import safetensors.torch
import onnx
from onnx.external_data_helper import convert_model_to_external_data

MODEL_DIR = "/data/models/image"

class CLIPVisionViT_Hidden(nn.Module):
    """OpenCLIP ViT-bigG/14 outputting hidden_states[-2] = [1,257,1280]."""
    def __init__(self, weights):
        super().__init__()
        self.hidden_size = 1280
        self.patch_size = 14
        self.num_patches = (224 // 14) ** 2   # 256
        self.num_positions = self.num_patches + 1  # 257

        # Patch embedding
        self.patch_embedding = nn.Conv2d(3, 1280, kernel_size=14, stride=14, bias=False)
        self.patch_embedding.weight.data = weights['vision_model.embeddings.patch_embedding.weight']

        # Class & position embeddings
        self.class_embedding = nn.Parameter(weights['vision_model.embeddings.class_embedding'])
        self.position_embedding = nn.Parameter(weights['vision_model.embeddings.position_embedding.weight'])

        # Pre-layernorm
        self.pre_layrnorm = nn.LayerNorm(1280, eps=1e-5)
        self.pre_layrnorm.weight.data = weights['vision_model.pre_layrnorm.weight']
        self.pre_layrnorm.bias.data = weights['vision_model.pre_layrnorm.bias']

        # Transformer layers (32)
        self.layers = nn.ModuleList()
        for i in range(32):
            layer = TransformerLayer(1280, weights, prefix=f'vision_model.encoder.layers.{i}')
            self.layers.append(layer)

        # Post-layernorm
        self.post_layernorm = nn.LayerNorm(1280, eps=1e-5)
        self.post_layernorm.weight.data = weights['vision_model.post_layernorm.weight']
        self.post_layernorm.bias.data = weights['vision_model.post_layernorm.bias']

        # NO visual_projection — we output raw hidden_states

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        # [B,3,224,224] -> [B,1280,16,16] -> [B,1280,256] -> [B,256,1280]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # CLS token
        cls_token = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        hidden_states = torch.cat([cls_token, patch_embeds], dim=1)  # [B,257,1280]
        hidden_states = hidden_states + self.position_embedding.unsqueeze(0)
        hidden_states = self.pre_layrnorm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        # Output ALL tokens: [B, 257, 1280] (used by IPAdapter Perceiver Resampler)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, weights, prefix):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm1.weight.data = weights[f'{prefix}.layer_norm1.weight']
        self.layer_norm1.bias.data = weights[f'{prefix}.layer_norm1.bias']
        self.self_attn = SelfAttention(hidden_size, weights, prefix)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm2.weight.data = weights[f'{prefix}.layer_norm2.weight']
        self.layer_norm2.bias.data = weights[f'{prefix}.layer_norm2.bias']
        self.mlp = MLP(hidden_size, weights, prefix)

    def forward(self, x):
        r = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = r + x
        r = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = r + x
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, weights, prefix):
        super().__init__()
        self.num_heads = 16
        self.head_dim = hidden_size // 16
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_proj.weight.data = weights[f'{prefix}.self_attn.q_proj.weight']
        self.q_proj.bias.data = weights[f'{prefix}.self_attn.q_proj.bias']
        self.k_proj.weight.data = weights[f'{prefix}.self_attn.k_proj.weight']
        self.k_proj.bias.data = weights[f'{prefix}.self_attn.k_proj.bias']
        self.v_proj.weight.data = weights[f'{prefix}.self_attn.v_proj.weight']
        self.v_proj.bias.data = weights[f'{prefix}.self_attn.v_proj.bias']
        self.out_proj.weight.data = weights[f'{prefix}.self_attn.out_proj.weight']
        self.out_proj.bias.data = weights[f'{prefix}.self_attn.out_proj.bias']

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, 16, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, 16, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, 16, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, hidden_size, weights, prefix):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size, bias=True)
        self.fc1.weight.data = weights[f'{prefix}.mlp.fc1.weight']
        self.fc1.bias.data = weights[f'{prefix}.mlp.fc1.bias']
        self.fc2.weight.data = weights[f'{prefix}.mlp.fc2.weight']
        self.fc2.bias.data = weights[f'{prefix}.mlp.fc2.bias']

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate='tanh')
        x = self.fc2(x)
        return x


def main():
    input_path = os.path.join(MODEL_DIR, "clip_vision_sd15.safetensors")
    output_onnx = os.path.join(MODEL_DIR, "clip_vision_vit_h_hidden.onnx")

    print(f"[INFO] Loading weights from {input_path}...")
    weights = safetensors.torch.load_file(input_path)
    print(f"[INFO] Loaded {len(weights)} tensors")

    print("[INFO] Building CLIP Vision model (hidden states output)...")
    model = CLIPVisionViT_Hidden(weights)
    model.eval()

    # Verify output shape
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
        print(f"[INFO] Output shape: {out.shape}  (expected [1, 257, 1280])")
        assert out.shape == (1, 257, 1280), f"Wrong output shape: {out.shape}"

    # Export to ONNX
    # PyTorch 2.4 automatically uses external data for large models (>2GB protobuf limit)
    print(f"[INFO] Exporting to ONNX: {output_onnx}")
    torch.onnx.export(
        model,
        dummy,
        output_onnx,
        input_names=['pixel_values'],
        output_names=['hidden_states'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'hidden_states': {0: 'batch_size'},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    # Reload and force-convert to combined external data file
    # This ensures we get a clean .onnx + .onnx.data pair (not individual weight files)
    print("[INFO] Converting to combined external data format...")
    model_proto = onnx.load(output_onnx, load_external_data=False)
    convert_model_to_external_data(
        model_proto,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_onnx) + ".data",
        convert_attribute=True,
    )
    onnx.save_model(
        model_proto,
        output_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_onnx) + ".data",
    )

    # Verify the exported model
    print("[INFO] Verifying ONNX export...")
    onnx.checker.check_model(output_onnx)
    print("[INFO] ONNX model verified OK")

    # Test with ONNX Runtime
    print("[INFO] Testing with ONNX Runtime...")
    import onnxruntime as ort
    session = ort.InferenceSession(output_onnx, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] Input: '{input_name}' shape: {session.get_inputs()[0].shape}")
    print(f"[INFO] Output: '{output_name}' shape: {session.get_outputs()[0].shape}")

    ort_out = session.run([output_name], {input_name: dummy.numpy()})[0]
    print(f"[INFO] ONNX Runtime output shape: {ort_out.shape}")
    assert ort_out.shape == (1, 257, 1280), f"Wrong ONNX shape: {ort_out.shape}"

    # Compare to PyTorch
    max_diff = (ort_out - out.numpy()).max()
    print(f"[INFO] Max diff PyTorch vs ONNX Runtime: {max_diff:.6f}")

    # File sizes
    onnx_size = os.path.getsize(output_onnx)
    data_size = os.path.getsize(output_onnx + ".data") if os.path.exists(output_onnx + ".data") else 0
    print(f"[INFO] {os.path.basename(output_onnx)}: {onnx_size / 1024:.0f} KB")
    print(f"[INFO] {os.path.basename(output_onnx)}.data: {data_size / 1024 / 1024:.1f} MB")
    print("[INFO] DONE — CLIP Vision ONNX with hidden states successfully regenerated!")


if __name__ == '__main__':
    main()
