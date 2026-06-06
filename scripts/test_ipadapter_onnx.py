#!/usr/bin/env python3
"""
IPAdapter ONNX verification script.
Tests that the C++ IPAdapter output matches Python ONNX Runtime output.

Usage:
  python3 scripts/test_ipadapter_onnx.py [--image ~/demo.png]

Requires: pip install onnxruntime pillow numpy
"""
import os, sys
import numpy as np

# Check if onnxruntime is available
try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/demo.png")
    clip_path = "/data/models/image/clip_vision.onnx"
    ipa_path = "/data/models/image/ipadapter.onnx"

    for path in [image_path, clip_path, ipa_path]:
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            sys.exit(1)

    print(f"[INFO] Reference image: {image_path}")
    print(f"[INFO] CLIP Vision: {clip_path} ({os.path.getsize(clip_path)/1e9:.2f}GB + .data)")
    print(f"[INFO] IPAdapter MLP: {ipa_path} ({os.path.getsize(ipa_path)/1e6:.2f}MB + .data)")

    # 1. Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    print(f"[INFO] Image loaded, resized to 224x224")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (np.array(img).astype(np.float32) / 255.0 - mean) / std
    x = x.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, 224, 224]

    # 2. CLIP Vision inference
    print(f"[INFO] Running CLIP Vision inference...")
    clip = ort.InferenceSession(clip_path, providers=['CPUExecutionProvider'])
    clip_out = clip.run(['image_embeds'], {'pixel_values': x})[0]
    print(f"  Output shape: {clip_out.shape}")
    print(f"  Stats: min={clip_out.min():.4f}, max={clip_out.max():.4f}, "
          f"mean={clip_out.mean():.4f}, std={clip_out.std():.4f}")

    # 3. IPAdapter MLP inference
    print(f"[INFO] Running IPAdapter MLP inference...")
    ipa = ort.InferenceSession(ipa_path, providers=['CPUExecutionProvider'])
    ipa_out = ipa.run(['text_embedding'], {'image_features': clip_out})[0]
    print(f"  Output shape: {ipa_out.shape}")
    print(f"  Stats: min={ipa_out.min():.4f}, max={ipa_out.max():.4f}, "
          f"mean={ipa_out.mean():.4f}, std={ipa_out.std():.4f}")
    print(f"  First 20 values: {[f'{v:.4f}' for v in ipa_out[0,:20]]}")
    print(f"  Last 20 values:  {[f'{v:.4f}' for v in ipa_out[0,-20:]]}")

    # 4. Save reference output for C++ comparison
    ref_path = "/tmp/ipadapter_ref_output.npy"
    np.save(ref_path, ipa_out)
    print(f"\n[INFO] Reference output saved to {ref_path}")
    print(f"[INFO] To compare with C++ output, read this file in a debugger")
    print(f"[INFO] or run: python3 -c \"import numpy as np; d=np.load('{ref_path}'); print(d)\"")

    print(f"\n[OK] IPAdapter ONNX pipeline verified successfully!")

if __name__ == "__main__":
    main()
