#!/usr/bin/env python3
"""
Convert ControlNet preprocessor models from PyTorch state_dict to TorchScript format.
After conversion, C++ code can use libtorch to load and run these models directly,
without Python runtime dependency.
"""

import os
import sys
import torch
import torch.nn as nn

MODEL_DIR = "/opt/image/model"

def patch_midas_vit():
    """Patch MiDaS vit module to use fixed-size unflatten for tracing."""
    from controlnet_aux.midas.midas import vit
    import controlnet_aux.midas.midas.dpt_depth as dpt_depth
    
    def patched_forward_vit(pretrained, x):
        b, c, h, w = x.shape
        
        glob = pretrained.model.forward_flex(x)
        
        layer_1 = pretrained.activations["1"]
        layer_2 = pretrained.activations["2"]
        layer_3 = pretrained.activations["3"]
        layer_4 = pretrained.activations["4"]
        
        layer_1 = pretrained.act_postprocess1[0:2](layer_1)
        layer_2 = pretrained.act_postprocess2[0:2](layer_2)
        layer_3 = pretrained.act_postprocess3[0:2](layer_3)
        layer_4 = pretrained.act_postprocess4[0:2](layer_4)
        
        # Use reshape instead of dynamic Unflatten
        # For 384x384 input with patch_size=16, this is 24x24
        patch_h = 24
        patch_w = 24
        
        if layer_1.ndim == 3:
            layer_1 = layer_1.reshape(b, -1, patch_h, patch_w)
        if layer_2.ndim == 3:
            layer_2 = layer_2.reshape(b, -1, patch_h, patch_w)
        if layer_3.ndim == 3:
            layer_3 = layer_3.reshape(b, -1, patch_h, patch_w)
        if layer_4.ndim == 3:
            layer_4 = layer_4.reshape(b, -1, patch_h, patch_w)
        
        layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
        layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
        layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
        layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)
        
        return layer_1, layer_2, layer_3, layer_4
    
    # Patch both the vit module and the dpt_depth module's cached reference
    vit.forward_vit = patched_forward_vit
    dpt_depth.forward_vit = patched_forward_vit
    print("Patched MiDaS forward_vit in both vit and dpt_depth modules")


def convert_midas_to_torchscript():
    """Convert MiDaS DPT Hybrid model to TorchScript."""
    print("=" * 60)
    print("Converting MiDaS DPT Hybrid to TorchScript...")
    print("=" * 60)
    
    input_path = os.path.join(MODEL_DIR, "dpt_hybrid-midas-501f0c75.pt")
    output_path = os.path.join(MODEL_DIR, "midas_dpt_hybrid.pt")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return False
    
    if os.path.exists(output_path):
        print(f"TorchScript model already exists: {output_path}")
        return True
    
    try:
        # Patch before any model creation
        patch_midas_vit()
        
        # Now import and load model
        from controlnet_aux.midas.api import MiDaSInference
        
        model = MiDaSInference(
            model_type='dpt_hybrid',
            model_path=input_path
        )
        model.eval()
        print("Loaded MiDaS DPT Hybrid model")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 384, 384)
        
        # Trace the model
        print("Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        
        # Save
        print("Saving TorchScript model...")
        traced_model.save(output_path)
        
        print(f"Successfully exported to: {output_path}")
        print(f"Output size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        return True
        
    except Exception as e:
        print(f"Error converting MiDaS: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_openpose_to_torchscript():
    """Convert OpenPose body model to TorchScript."""
    print("\n" + "=" * 60)
    print("Converting OpenPose Body Model to TorchScript...")
    print("=" * 60)
    
    input_path = os.path.join(MODEL_DIR, "body_pose_model.pth")
    output_path = os.path.join(MODEL_DIR, "openpose_body.pt")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return False
    
    if os.path.exists(output_path):
        print(f"TorchScript model already exists: {output_path}")
        return True
    
    try:
        from controlnet_aux.open_pose.body import Body
        
        body = Body(input_path)
        model = body.model
        model.eval()
        print("Loaded OpenPose body model")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 368, 368)
        
        # Trace the model
        print("Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        
        # Save
        print("Saving TorchScript model...")
        traced_model.save(output_path)
        
        print(f"Successfully exported to: {output_path}")
        print(f"Output size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        return True
        
    except Exception as e:
        print(f"Error converting OpenPose: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("ControlNet Preprocessor Model Converter")
    print("Converts .pt/.pth models to TorchScript format for C++ inference")
    print()
    
    # Convert models
    midas_ok = convert_midas_to_torchscript()
    openpose_ok = convert_openpose_to_torchscript()
    
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"MiDaS Depth:     {'SUCCESS' if midas_ok else 'FAILED'}")
    print(f"OpenPose Body:   {'SUCCESS' if openpose_ok else 'FAILED'}")
    print()
    
    if not midas_ok or not openpose_ok:
        sys.exit(1)
    else:
        print("All conversions completed successfully!")
        print("C++ code can now load these models with libtorch (no Python needed).")
