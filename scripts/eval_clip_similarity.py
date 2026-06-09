#!/usr/bin/env python3
"""
CLIP image-image similarity evaluation.

Computes cosine similarity between a reference image and one or more
candidate images using the same CLIP Vision encoder that my-img uses
for IPAdapter preprocessing (/data/models/image/clip_vision.onnx).

Usage:
    # Single comparison
    python3 scripts/eval_clip_similarity.py \
        --reference /path/to/ref.png \
        --candidate /path/to/gen1.png /path/to/gen2.png

    # Directory mode (compares reference to every image in a directory)
    python3 scripts/eval_clip_similarity.py \
        --reference /path/to/ref.png \
        --dir /path/to/outputs/

    # Head-to-head mode (two directories, same reference)
    python3 scripts/eval_clip_similarity.py \
        --reference /path/to/ref.png \
        --dir-a /path/to/dit_outputs/ \
        --dir-b /path/to/unet_outputs/

    # Save JSON report
    python3 scripts/eval_clip_similarity.py ... --output report.json

Environment:
    CLIP_VISION_MODEL  - Override default clip_vision.onnx path
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
    raise exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    print("[ERROR] Pillow not installed. Run: pip install Pillow")
    raise exc

DEFAULT_CLIP_VISION = "/data/models/image/clip_vision.onnx"


def get_clip_vision_path() -> str:
    path = os.environ.get("CLIP_VISION_MODEL", DEFAULT_CLIP_VISION)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CLIP Vision model not found: {path}")
    return path


def load_session(model_path: str) -> ort.InferenceSession:
    # Prefer CUDA if available, otherwise CPU.
    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(model_path, providers=providers)
    used = sess.get_providers()[0]
    print(f"[INFO] CLIP Vision provider: {used}")
    return sess


def preprocess(image_path: str) -> np.ndarray:
    """Load image, resize to 224x224, apply CLIP normalization."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)

    # OpenAI CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, 224, 224]
    return arr


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return embeddings / norms


def compute_embeddings(session: ort.InferenceSession, image_paths: List[str]) -> np.ndarray:
    """Batch-compute CLIP embeddings for a list of images."""
    if not image_paths:
        return np.zeros((0, 1024), dtype=np.float32)

    batch = []
    for path in image_paths:
        batch.append(preprocess(path))
    batch = np.concatenate(batch, axis=0)  # [N, 3, 224, 224]

    outputs = session.run(["image_embeds"], {"pixel_values": batch})[0]
    return normalize(outputs)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one vector and many vectors."""
    # a: [D], b: [N, D] -> [N]
    return np.clip(np.dot(b, a), -1.0, 1.0)


def gather_images(paths: List[str]) -> List[str]:
    images = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            images.extend(sorted(
                str(x) for x in path.iterdir()
                if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            ))
        elif path.is_file():
            images.append(str(path))
        else:
            print(f"[WARN] Skipping non-existent path: {p}")
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP image-image similarity."
    )
    parser.add_argument("--reference", "-r", required=True,
                        help="Path to reference image.")
    parser.add_argument("--candidate", "-c", nargs="+", default=None,
                        help="One or more candidate images or directories.")
    parser.add_argument("--dir", "-d", default=None,
                        help="Directory of candidate images.")
    parser.add_argument("--dir-a", default=None,
                        help="Directory A for head-to-head comparison.")
    parser.add_argument("--dir-b", default=None,
                        help="Directory B for head-to-head comparison.")
    parser.add_argument("--output", "-o", default=None,
                        help="Optional JSON file to write report.")
    parser.add_argument("--clip-vision", default=None,
                        help="Path to clip_vision.onnx (default: env or /data/models/image/clip_vision.onnx).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-file embeddings stats.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    clip_path = args.clip_vision or get_clip_vision_path()
    print(f"[INFO] CLIP Vision model: {clip_path}")
    print(f"[INFO] Reference image:   {args.reference}")

    if not os.path.exists(args.reference):
        print(f"[ERROR] Reference image not found: {args.reference}")
        return 1

    session = load_session(clip_path)

    # Determine candidates
    candidates: List[str] = []
    head_to_head = False
    if args.dir_a and args.dir_b:
        candidates_a = gather_images([args.dir_a])
        candidates_b = gather_images([args.dir_b])
        candidates = candidates_a + candidates_b
        head_to_head = True
        labels = ["A"] * len(candidates_a) + ["B"] * len(candidates_b)
    elif args.candidate:
        candidates = gather_images(args.candidate)
        labels = [""] * len(candidates)
    elif args.dir:
        candidates = gather_images([args.dir])
        labels = [""] * len(candidates)
    else:
        print("[ERROR] Must specify --candidate, --dir, or both --dir-a and --dir-b.")
        return 1

    if not candidates:
        print("[ERROR] No candidate images found.")
        return 1

    # Compute embeddings
    ref_emb = compute_embeddings(session, [args.reference])[0]
    cand_embs = compute_embeddings(session, candidates)
    similarities = cosine_similarity(ref_emb, cand_embs)

    # Build report
    report = {
        "reference": args.reference,
        "clip_vision_model": clip_path,
        "reference_embedding_norm": float(np.linalg.norm(ref_emb)),
        "candidates": [],
    }

    if head_to_head:
        report["comparison"] = {
            "dir_a": args.dir_a,
            "dir_b": args.dir_b,
            "count_a": len(candidates_a),
            "count_b": len(candidates_b),
        }

    print("\n" + "=" * 70)
    print(f"{'Label':<6} {'Similarity':<12} {'File'}")
    print("=" * 70)

    group_sims: dict = {}
    for label, path, sim in zip(labels, candidates, similarities):
        entry = {
            "path": path,
            "similarity": float(sim),
            "label": label,
        }
        report["candidates"].append(entry)

        group_sims.setdefault(label, []).append(sim)
        print(f"{label:<6} {sim:>10.4f}    {path}")

    print("=" * 70)

    # Summary statistics
    if head_to_head:
        sims_a = np.array(group_sims.get("A", []))
        sims_b = np.array(group_sims.get("B", []))
        print(f"\nGroup A mean: {sims_a.mean():.4f}  std: {sims_a.std():.4f}  n={len(sims_a)}")
        print(f"Group B mean: {sims_b.mean():.4f}  std: {sims_b.std():.4f}  n={len(sims_b)}")
        if len(sims_a) > 0 and len(sims_b) > 0:
            delta = sims_b.mean() - sims_a.mean()
            print(f"Delta (B - A): {delta:+.4f}")
            report["summary"] = {
                "A": {"mean": float(sims_a.mean()), "std": float(sims_a.std()), "n": len(sims_a)},
                "B": {"mean": float(sims_b.mean()), "std": float(sims_b.std()), "n": len(sims_b)},
                "delta_B_minus_A": float(delta),
            }
    else:
        all_sims = np.array(list(group_sims.values())[0])
        print(f"\nMean similarity: {all_sims.mean():.4f}")
        print(f"Std similarity:  {all_sims.std():.4f}")
        print(f"Min similarity:  {all_sims.min():.4f}")
        print(f"Max similarity:  {all_sims.max():.4f}")
        report["summary"] = {
            "mean": float(all_sims.mean()),
            "std": float(all_sims.std()),
            "min": float(all_sims.min()),
            "max": float(all_sims.max()),
            "n": len(all_sims),
        }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Report saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
