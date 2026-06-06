#!/usr/bin/env python3
"""
Convert CLIP Vision safetensors (OpenCLIP ViT-bigG/14) to ONNX format.

Model: clip_vision_sd15.safetensors
Architecture: OpenCLIP ViT-bigG/14 (patch_size=14, hidden_size=1280, 32 layers)
Output: image embeddings [1, 1024] (after visual_projection)
"""

import os
import sys
import torch
import torch.nn as nn
import safetensors.torch
import argparse

class CLIPVisionViT(nn.Module):
    """OpenCLIP ViT-bigG/14 for image encoding."""
    
    def __init__(self, weights):
        super().__init__()
        self.hidden_size = 1280
        self.patch_size = 14
        self.num_patches = (224 // 14) ** 2  # 256
        self.num_positions = self.num_patches + 1  # 257 (CLS + patches)
        
        # Patch embedding: Conv2d(in=3, out=1280, kernel=14, stride=14)
        self.patch_embedding = nn.Conv2d(3, 1280, kernel_size=14, stride=14, bias=False)
        self.patch_embedding.weight.data = weights['vision_model.embeddings.patch_embedding.weight']
        
        # Class embedding
        self.class_embedding = nn.Parameter(weights['vision_model.embeddings.class_embedding'])
        
        # Position embedding
        self.position_embedding = nn.Parameter(weights['vision_model.embeddings.position_embedding.weight'])
        
        # Pre-layernorm
        self.pre_layrnorm = nn.LayerNorm(1280, eps=1e-5)
        self.pre_layrnorm.weight.data = weights['vision_model.pre_layrnorm.weight']
        self.pre_layrnorm.bias.data = weights['vision_model.pre_layrnorm.bias']
        
        # Transformer encoder layers
        self.layers = nn.ModuleList()
        for i in range(32):
            layer = TransformerLayer(1280, weights, prefix=f'vision_model.encoder.layers.{i}')
            self.layers.append(layer)
        
        # Post-layernorm
        self.post_layernorm = nn.LayerNorm(1280, eps=1e-5)
        self.post_layernorm.weight.data = weights['vision_model.post_layernorm.weight']
        self.post_layernorm.bias.data = weights['vision_model.post_layernorm.bias']
        
        # Visual projection: 1280 -> 1024 (matching IPAdapter input)
        self.visual_projection = nn.Linear(1280, 1024, bias=False)
        self.visual_projection.weight.data = weights['visual_projection.weight']
    
    def forward(self, pixel_values):
        """Encode image to CLIP embedding.
        
        Args:
            pixel_values: [B, 3, 224, 224] normalized image
        Returns:
            image_embeds: [B, 1024] projected CLS token
        """
        B = pixel_values.shape[0]
        
        # Patch embedding: [B, 3, 224, 224] -> [B, 1280, 16, 16]
        patch_embeds = self.patch_embedding(pixel_values)
        # Flatten: [B, 1280, 16, 16] -> [B, 1280, 256]
        patch_embeds = patch_embeds.flatten(2)
        # Transpose: [B, 1280, 256] -> [B, 256, 1280]
        patch_embeds = patch_embeds.transpose(1, 2)
        
        # Add CLS token: [B, 1, 1280]
        cls_token = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        
        # Concat: [B, 257, 1280]
        hidden_states = torch.cat([cls_token, patch_embeds], dim=1)
        
        # Add position embedding
        hidden_states = hidden_states + self.position_embedding.unsqueeze(0)
        
        # Pre-layernorm
        hidden_states = self.pre_layrnorm(hidden_states)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Post-layernorm
        hidden_states = self.post_layernorm(hidden_states)
        
        # Extract CLS token and project: [B, 1280] -> [B, 1024]
        cls_output = hidden_states[:, 0, :]
        image_embeds = self.visual_projection(cls_output)
        
        return image_embeds


class TransformerLayer(nn.Module):
    """Single ViT encoder layer with attention and MLP."""
    
    def __init__(self, hidden_size, weights, prefix):
        super().__init__()
        self.hidden_size = hidden_size
        
        # LayerNorm 1
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm1.weight.data = weights[f'{prefix}.layer_norm1.weight']
        self.layer_norm1.bias.data = weights[f'{prefix}.layer_norm1.bias']
        
        # Self-attention
        self.self_attn = SelfAttention(hidden_size, weights, prefix)
        
        # LayerNorm 2
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm2.weight.data = weights[f'{prefix}.layer_norm2.weight']
        self.layer_norm2.bias.data = weights[f'{prefix}.layer_norm2.bias']
        
        # MLP
        self.mlp = MLP(hidden_size, weights, prefix)
    
    def forward(self, x):
        # Self-attention with residual
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, hidden_size, weights, prefix):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = 16  # ViT-bigG uses 16 heads (1280 / 80 = 16, but usually 1280/16=16 heads with 80 dim each)
        # Actually for ViT-bigG: 1280 dim, 16 heads, 80 dim per head
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
    """Two-layer MLP with GELU."""
    
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
    parser = argparse.ArgumentParser(description='Convert CLIP Vision to ONNX')
    parser.add_argument('--input', default='/data/models/image/clip_vision_sd15.safetensors',
                       help='Input safetensors path')
    parser.add_argument('--output', default='/data/models/image/clip_vision.onnx',
                       help='Output ONNX path')
    parser.add_argument('--opset', type=int, default=17,
                       help='ONNX opset version')
    args = parser.parse_args()
    
    print(f"[INFO] Loading weights from {args.input}...")
    weights = safetensors.torch.load_file(args.input)
    print(f"[INFO] Loaded {len(weights)} tensors")
    
    print("[INFO] Building CLIP Vision model...")
    model = CLIPVisionViT(weights)
    model.eval()
    
    # Verify with random input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"[INFO] Test output shape: {output.shape}")
        print(f"[INFO] Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
    
    # Export to ONNX
    print(f"[INFO] Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=['pixel_values'],
        output_names=['image_embeds'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'image_embeds': {0: 'batch_size'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
    )
    
    # Verify ONNX
    print("[INFO] Verifying ONNX export...")
    import onnx
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print(f"[INFO] ONNX model saved to {args.output}")
    print(f"[INFO] ONNX IR version: {onnx_model.ir_version}")
    
    # Test with ONNX Runtime
    print("[INFO] Testing with ONNX Runtime...")
    import onnxruntime as ort
    session = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] Input: {input_name}, shape: {session.get_inputs()[0].shape}")
    print(f"[INFO] Output: {output_name}, shape: {session.get_outputs()[0].shape}")
    
    # Compare outputs
    ort_input = {input_name: dummy_input.numpy()}
    ort_output = session.run([output_name], ort_input)[0]
    torch_output = output.numpy()
    
    max_diff = np.max(np.abs(ort_output - torch_output))
    print(f"[INFO] Max diff between PyTorch and ONNX Runtime: {max_diff:.6f}")
    
    file_size = os.path.getsize(args.output) / (1024 * 1024 * 1024)
    print(f"[INFO] File size: {file_size:.2f} GB")


if __name__ == '__main__':
    import numpy as np
    main()
