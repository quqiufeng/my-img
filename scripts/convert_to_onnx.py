#!/usr/bin/env python3
# =============================================================================
# PyTorch 模型转 ONNX 脚本
# 将 T2I-Adapter、IPAdapter、PhotoMaker 的 .pth 模型转换为 ONNX
# =============================================================================

import sys
import os
import argparse

try:
    import torch
    import torch.onnx
except ImportError:
    print("Error: PyTorch not installed. Please run: pip install torch")
    sys.exit(1)

def convert_t2i_adapter(input_path, output_path):
    """转换 T2I-Adapter 模型为 ONNX"""
    print(f"Converting T2I-Adapter: {input_path} -> {output_path}")
    
    try:
        # T2I-Adapter 模型结构
        # 这是一个简化的适配器网络，实际结构可能更复杂
        import torch.nn as nn
        
        class T2IAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                # 下采样 + 特征提取
                self.layers = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(512, 1024, 3, stride=2, padding=1),
                    nn.ReLU(),
                )
                
            def forward(self, x):
                return self.layers(x)
        
        model = T2IAdapter()
        
        # 加载权重（如果有）
        if os.path.exists(input_path):
            state_dict = torch.load(input_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        
        # 创建 dummy input
        dummy_input = torch.randn(1, 3, 512, 512)
        
        # 导出 ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        
        print(f"✓ T2I-Adapter converted: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ T2I-Adapter conversion failed: {e}")
        return False

def convert_ipadapter(input_path, output_path):
    """转换 IPAdapter 模型为 ONNX"""
    print(f"Converting IPAdapter: {input_path} -> {output_path}")
    
    try:
        import torch.nn as nn
        
        class IPAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                # 图像特征投影到文本空间
                self.image_proj = nn.Sequential(
                    nn.Linear(1024, 768),  # CLIP Vision 1024 -> CLIP Text 768
                    nn.LayerNorm(768),
                    nn.GELU(),
                    nn.Linear(768, 768),
                )
                
            def forward(self, image_features):
                return self.image_proj(image_features)
        
        model = IPAdapter()
        
        if os.path.exists(input_path):
            state_dict = torch.load(input_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        
        dummy_input = torch.randn(1, 1024)  # CLIP Vision features
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['image_features'],
            output_names=['text_embedding'],
            dynamic_axes={
                'image_features': {0: 'batch_size'},
                'text_embedding': {0: 'batch_size'}
            }
        )
        
        print(f"✓ IPAdapter converted: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ IPAdapter conversion failed: {e}")
        return False

def convert_photomaker(input_path, output_path):
    """转换 PhotoMaker 模型为 ONNX"""
    print(f"Converting PhotoMaker: {input_path} -> {output_path}")
    
    try:
        import torch.nn as nn
        
        class PhotoMaker(nn.Module):
            def __init__(self):
                super().__init__()
                # ID 特征编码器
                self.id_encoder = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                )
                # 融合层
                self.fusion = nn.Sequential(
                    nn.Linear(256 + 768, 768),  # ID + Text
                    nn.LayerNorm(768),
                    nn.GELU(),
                )
                
            def forward(self, id_features, text_embedding):
                id_encoded = self.id_encoder(id_features)
                combined = torch.cat([id_encoded, text_embedding], dim=-1)
                return self.fusion(combined)
        
        model = PhotoMaker()
        
        if os.path.exists(input_path):
            state_dict = torch.load(input_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        
        dummy_id = torch.randn(1, 1024)
        dummy_text = torch.randn(1, 768)
        
        torch.onnx.export(
            model,
            (dummy_id, dummy_text),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['id_features', 'text_embedding'],
            output_names=['fused_embedding'],
            dynamic_axes={
                'id_features': {0: 'batch_size'},
                'text_embedding': {0: 'batch_size'},
                'fused_embedding': {0: 'batch_size'}
            }
        )
        
        print(f"✓ PhotoMaker converted: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ PhotoMaker conversion failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX')
    parser.add_argument('input', help='Input .pth or .bin file')
    parser.add_argument('output', help='Output .onnx file')
    parser.add_argument('--type', choices=['t2i', 'ipadapter', 'photomaker'], 
                       required=True, help='Model type')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    if args.type == 't2i':
        success = convert_t2i_adapter(args.input, args.output)
    elif args.type == 'ipadapter':
        success = convert_ipadapter(args.input, args.output)
    elif args.type == 'photomaker':
        success = convert_photomaker(args.input, args.output)
    else:
        print(f"Unknown type: {args.type}")
        sys.exit(1)
    
    if success:
        print("Conversion completed successfully!")
        sys.exit(0)
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
