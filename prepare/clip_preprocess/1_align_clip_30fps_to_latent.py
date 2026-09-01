#!/usr/bin/env python3
"""
Align 30 fps frame-level CLIP embeddings to the TAE latent temporal grid.

Each latent i is influenced by original frames from 4i-16 to 4i+4 (20 frames).
Average CLIP over that window → one 512-d vector per latent.

Input:  clip_enc_single_30fps/<id>.npy     shape (T_30, 512) at 30 fps
Output: clip_enc_latent_aligned/<id>.npy   shape (L, 512)

This output directory is DEFAULT_DATA_DIR in models/clip_autoencoder/2_train.py.

L is taken from datasets/motions/t2m_latents/<id>.npy (accepts (L, 16) or (16, L)).

Usage (from repo root):
  python prepare/clip_preprocess/1_align_clip_30fps_to_latent.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DOWNSAMPLE_FACTOR = 4
LATENT_FRAME_OFFSET_LEFT = 16
LATENT_FRAME_OFFSET_RIGHT = 4


def frame_window_for_latent(i: int, downsample: int = DOWNSAMPLE_FACTOR) -> tuple[int, int]:
    """Return (start_frame, end_frame_exclusive) for latent i."""
    start = max(0, downsample * i - LATENT_FRAME_OFFSET_LEFT)
    end_excl = downsample * i + LATENT_FRAME_OFFSET_RIGHT + 1
    return start, end_excl


def latent_temporal_length(arr: np.ndarray) -> int | None:
    """T-length of a 16-d TAE latent array stored as (L, 16) or (16, L)."""
    if arr.ndim != 2:
        return None
    h, w = arr.shape
    if h == 16:
        return int(w)
    if w == 16:
        return int(h)
    return None


def align_clip_to_latent(clip_30fps: np.ndarray, target_L: int | None = None) -> np.ndarray:
    """Aggregate (T_30, 512) to (L, 512) by averaging the causal TAE window."""
    T, dim = clip_30fps.shape
    if T == 0:
        return np.zeros((target_L or 0, dim), dtype=np.float32)
    L = target_L if target_L is not None else (T + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR
    out = np.zeros((L, dim), dtype=np.float32)
    for i in range(L):
        start, end_excl = frame_window_for_latent(i)
        start = max(0, min(start, T))
        end_excl = min(end_excl, T)
        if start >= end_excl:
            start = min(start, T - 1)
            end_excl = start + 1
        window = clip_30fps[start:end_excl]
        out[i] = window.mean(axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align 30fps CLIP to latent grid (4i-16 to 4i+4, average)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="(T_30, 512) npy folder (default: datasets/embeddings/temp_clip/clip_enc_single_30fps)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder (default: datasets/embeddings/temp_clip/clip_enc_latent_aligned)",
    )
    parser.add_argument("--limit", type=int, default=0, help="If > 0, process only the first N files")
    parser.add_argument(
        "--latent_dir",
        type=str,
        default="datasets/motions/t2m_latents",
        help="TAE latents used only to read L per id. Accepts (L, 16) or (16, L).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else REPO_ROOT / "datasets" / "embeddings" / "temp_clip" / "clip_enc_single_30fps"
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "datasets" / "embeddings" / "temp_clip" / "clip_enc_latent_aligned"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    if not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    latent_dir = Path(args.latent_dir)
    if not latent_dir.is_absolute():
        latent_dir = REPO_ROOT / latent_dir
    if not latent_dir.is_dir():
        print(f"Error: latent dir not found: {latent_dir}")
        print("Run python prepare/download_streamer272_data.py first.")
        sys.exit(1)

    latent_L_by_base: dict[str, int] = {}
    bad_shapes = 0
    for root, _, names in os.walk(latent_dir):
        for name in names:
            if not name.endswith(".npy"):
                continue
            arr = np.load(os.path.join(root, name))
            L = latent_temporal_length(arr)
            if L is None:
                bad_shapes += 1
                continue
            latent_L_by_base[name[:-4]] = L

    if not latent_L_by_base:
        print(f"Error: no usable latent files in {latent_dir}. Expected (L, 16) or (16, L).")
        sys.exit(1)

    print(f"  Using L from {len(latent_L_by_base)} latent files in {latent_dir}")
    if bad_shapes:
        print(f"  Skipped {bad_shapes} latent files with unexpected shape")

    files: list[tuple[str, Path, str]] = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if name.endswith(".npy"):
                full = Path(root) / name
                rel = os.path.relpath(full, input_dir)
                base = os.path.splitext(os.path.basename(rel))[0]
                files.append((rel, full, base))
    files.sort(key=lambda x: x[0])

    if args.limit > 0:
        files = files[: args.limit]

    n = len(files)
    print(f"Aligning {n} .npy files from 30fps frame-level to latent grid")
    print("  Window: latent i <- mean(frames [4i-16, 4i+4])")
    print(f"  input:  {input_dir}")
    print(f"  output: {output_dir}")
    print()

    skipped = 0
    for i, (rel, src_path, base) in enumerate(files):
        clip = np.load(src_path)
        if clip.ndim != 2:
            print(f"  [{i+1}/{n}] skip {rel}: expected 2D, got shape {clip.shape}")
            continue
        T, dim = clip.shape
        if dim != 512:
            print(f"  [{i+1}/{n}] skip {rel}: expected 512 dims, got {dim}")
            continue
        target_L = latent_L_by_base.get(base)
        if target_L is None:
            print(f"  [{i+1}/{n}] skip {rel}: no latent length for '{base}' in {latent_dir}")
            skipped += 1
            continue
        aligned = align_clip_to_latent(clip.astype(np.float32), target_L=target_L)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, aligned)
        if (i + 1) % 2000 == 0 or i == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {rel}  (T={T}, 512) -> (L={aligned.shape[0]}, 512)")

    print(f"\nDone. Wrote {n - skipped} files to {output_dir} and skipped {skipped} files.")


if __name__ == "__main__":
    main()
