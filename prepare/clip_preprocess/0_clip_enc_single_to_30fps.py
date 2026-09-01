#!/usr/bin/env python3
"""
Resample clip_enc_single (T, 512) from 20 fps to 30 fps → clip_enc_single_30fps.

Nearest-neighbor in time (no interpolation): these are discrete label embeddings.

Input:  datasets/embeddings/temp_clip/clip_enc_single/<id>.npy      (T_20, 512)
Output: datasets/embeddings/temp_clip/clip_enc_single_30fps/<id>.npy (T_30, 512)
        T_30 = round(T_20 * 30/20)

Usage (from repo root):
  python prepare/clip_preprocess/0_clip_enc_single_to_30fps.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def resample_clip_to_target_length(clip: np.ndarray, target_T: int) -> np.ndarray:
    """Resample (T_in, D) to (target_T, D) via nearest-neighbor along axis 0."""
    T_in, dim = clip.shape
    if T_in == 0:
        return np.zeros((target_T, dim), dtype=clip.dtype)
    if target_T == 0:
        return np.zeros((0, dim), dtype=clip.dtype)
    if T_in == 1:
        return np.broadcast_to(clip, (target_T, dim)).copy()
    old_indices_float = np.linspace(0, T_in - 1, target_T, dtype=np.float64)
    nearest_indices = np.round(old_indices_float).astype(np.int64)
    nearest_indices = np.clip(nearest_indices, 0, T_in - 1)
    return clip[nearest_indices].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample clip_enc_single from 20 fps to 30 fps")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input folder with .npy files (default: datasets/embeddings/temp_clip/clip_enc_single)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder (default: datasets/embeddings/temp_clip/clip_enc_single_30fps)",
    )
    parser.add_argument("--fps_in", type=int, default=20)
    parser.add_argument("--fps_out", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="If > 0, process only the first N files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else REPO_ROOT / "datasets" / "embeddings" / "temp_clip" / "clip_enc_single"
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "datasets" / "embeddings" / "temp_clip" / "clip_enc_single_30fps"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    if not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}")
        print("Run clip_encoder.py first, or pass --input_dir.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    files: list[tuple[str, Path]] = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if name.endswith(".npy"):
                full = Path(root) / name
                rel = os.path.relpath(full, input_dir)
                files.append((rel, full))
    files.sort(key=lambda x: x[0])

    if args.limit > 0:
        files = files[: args.limit]

    n = len(files)
    print(f"Resampling {n} .npy files from {args.fps_in} fps to {args.fps_out} fps")
    print(f"  input:  {input_dir}")
    print(f"  output: {output_dir}")
    print()

    for i, (rel, src_path) in enumerate(files):
        clip = np.load(src_path)
        if clip.ndim != 2:
            print(f"  [{i+1}/{n}] skip {rel}: expected 2D, got shape {clip.shape}")
            continue
        T_in, dim = clip.shape
        if dim != 512:
            print(f"  [{i+1}/{n}] skip {rel}: expected 512 dims, got {dim}")
            continue
        T_out = int(round(T_in * args.fps_out / args.fps_in))
        if T_out < 1:
            T_out = 1
        resampled = resample_clip_to_target_length(clip.astype(np.float32), T_out)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, resampled)
        if (i + 1) % 2000 == 0 or i == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {rel}  ({T_in}, 512) -> ({T_out}, 512)")

    print(f"\nDone. Wrote {n} files to {output_dir}")


if __name__ == "__main__":
    main()
