#!/usr/bin/env python3
"""
Concatenate TAE motion latents with AE-16 CLIP latents → (32, L) training files.

Motion: datasets/motions/t2m_latents/<id>.npy                 (L, 16) or (16, L)
CLIP:   datasets/embeddings/temp_clip/clip_enc_latent_aligned_ae16/<id>.npy  (L, 16)
Output: datasets/motions/t2m_latent_frame_text_aligned/<id>.npy  (32, L)
        dims 0:16 motion, dims 16:32 text. Only IDs present in both sources
        with matching L are written.

Usage (from repo root):
  python prepare/clip_preprocess/4_build_frame_text_aligned.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LATENT_DIR = REPO_ROOT / "datasets" / "motions" / "t2m_latents"
DEFAULT_CLIP_DIR = (
    REPO_ROOT / "datasets" / "embeddings" / "temp_clip" / "clip_enc_latent_aligned_ae16"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "motions" / "t2m_latent_frame_text_aligned"


def collect_npy_basenames(root_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root_dir.is_dir():
        return out
    for dirpath, _, names in os.walk(root_dir):
        for name in names:
            if name.endswith(".npy"):
                base = name[:-4]
                full = Path(dirpath) / name
                if base not in out:
                    out[base] = full
    return out


def to_channel_first_16(arr: np.ndarray) -> np.ndarray | None:
    """Return (16, L) from (16, L) or (L, 16)."""
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 16:
        return arr
    if arr.shape[1] == 16:
        return arr.T
    return None


def resolve_dir(path: str | None, default: Path) -> Path:
    p = Path(path) if path else default
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build t2m_latent_frame_text_aligned: concat motion (16,L) + CLIP AE16 (16,L) → (32,L)"
    )
    parser.add_argument("--latent_dir", type=str, default=None)
    parser.add_argument("--clip_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    latent_dir = resolve_dir(args.latent_dir, DEFAULT_LATENT_DIR)
    clip_dir = resolve_dir(args.clip_dir, DEFAULT_CLIP_DIR)
    output_dir = resolve_dir(args.output_dir, DEFAULT_OUTPUT_DIR)

    if not latent_dir.is_dir():
        print(f"ERROR: Latent dir not found: {latent_dir}")
        sys.exit(1)
    if not clip_dir.is_dir():
        print(f"ERROR: Clip dir not found: {clip_dir}")
        sys.exit(1)

    latent_files = collect_npy_basenames(latent_dir)
    clip_files = collect_npy_basenames(clip_dir)
    common = sorted(set(latent_files.keys()) & set(clip_files.keys()))

    print(f"t2m_latents:                 {len(latent_files)} files")
    print(f"clip_enc_latent_aligned_ae16: {len(clip_files)} files")
    print(f"Matching IDs:                {len(common)}")
    print(f"Output:                      {output_dir}")
    print()

    if not common:
        print("No matching IDs. Encode CLIP with 3_encode_all.py and TAE latents first.")
        return

    if args.limit > 0:
        common = common[: args.limit]
        print(f"Limited to first {len(common)} IDs")
        print()

    os.makedirs(output_dir, exist_ok=True)

    written = 0
    skipped = 0
    length_mismatch = 0
    errors = 0

    for motion_id in tqdm(common, desc="Building frame-text-aligned"):
        out_path = output_dir / f"{motion_id}.npy"
        if out_path.is_file() and not args.overwrite:
            skipped += 1
            continue

        try:
            motion = to_channel_first_16(np.load(latent_files[motion_id]))
            clip = to_channel_first_16(np.load(clip_files[motion_id]))
            if motion is None:
                errors += 1
                tqdm.write(f"Skip {motion_id}: latent not (L, 16) or (16, L)")
                continue
            if clip is None:
                errors += 1
                tqdm.write(f"Skip {motion_id}: clip not (L, 16) or (16, L)")
                continue
            if motion.shape[1] != clip.shape[1]:
                length_mismatch += 1
                tqdm.write(
                    f"Skip {motion_id}: length mismatch L_motion={motion.shape[1]} L_clip={clip.shape[1]}"
                )
                continue

            combined = np.concatenate(
                [motion.astype(np.float32), clip.astype(np.float32)], axis=0
            )
            if written == 0:
                print(f"motion.shape: {motion.shape}, clip.shape: {clip.shape}")
                print(f"combined.shape: {combined.shape}")
            np.save(out_path, combined)
            written += 1
        except Exception as e:
            errors += 1
            tqdm.write(f"Error {motion_id}: {e}")

    print()
    print("Summary:")
    print(f"  Written:          {written}")
    print(f"  Skipped (exist): {skipped}")
    print(f"  Length mismatch: {length_mismatch}")
    print(f"  Errors:          {errors}")
    print("  Output shape:    (32, L) — first 16 motion latent, next 16 CLIP AE")
    print(f"  Output dir:     {output_dir}")


if __name__ == "__main__":
    main()
