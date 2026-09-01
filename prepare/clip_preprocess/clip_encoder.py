#!/usr/bin/env python3
"""
CLIP-encode per-frame action labels to [T, 512] arrays.

This repo does not include BABEL annotations. You must obtain BABEL from
https://babel.is.tue.mpg.de/ and convert it to the JSON format below.

Expected input (one file per HumanML3D id):
    <annotation_dir>/000002.json

    {
      "frame_label": [
        ["walk"],
        ["walk", "forward"]
      ]
    }

- Filename stem = motion id (written as <id>.npy).
- frame_label[t] is a non-empty list of strings for that 20 fps frame.
- Default --multi avg: mean CLIP embedding of all labels on the frame.
- --multi single: first label only.

Output:
    datasets/embeddings/temp_clip/clip_enc_single/<id>.npy   shape [T_20, 512]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ANNOTATION_DIR = REPO_ROOT / "datasets" / "embeddings" / "temp_clip" / "babel_frame_labels"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "embeddings" / "temp_clip" / "clip_enc_single"


class ClipEncoder(nn.Module):
    def __init__(self, clip_version: str, device: torch.device):
        super().__init__()
        self.device = device
        model, _preprocess = clip.load(clip_version, device="cpu", jit=False)
        clip.model.convert_weights(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.clip_model = model.to(device)

    def encode_text(self, raw_text: list[str]) -> torch.Tensor:
        tokens = clip.tokenize(raw_text, truncate=True).to(self.device)
        return self.clip_model.encode_text(tokens).float()

    def forward(self, text: str) -> torch.Tensor:
        return self.encode_text([text])


def _load_frame_labels(path: Path) -> list[list[str]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "frame_label" not in data:
        raise ValueError(f"{path.name}: expected a JSON object with key 'frame_label'")
    frames = data["frame_label"]
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError(f"{path.name}: 'frame_label' must be a non-empty list")
    cleaned: list[list[str]] = []
    for t, labels in enumerate(frames):
        if not isinstance(labels, list) or len(labels) == 0:
            raise ValueError(
                f"{path.name}: frame_label[{t}] must be a non-empty list of strings"
            )
        strs = [str(x) for x in labels if str(x).strip()]
        if not strs:
            raise ValueError(
                f"{path.name}: frame_label[{t}] has no non-empty label strings"
            )
        cleaned.append(strs)
    return cleaned


def _embed_frames(
    model: ClipEncoder,
    frames: list[list[str]],
    multi: str,
) -> np.ndarray:
    gt_enc: list[torch.Tensor] = []
    for labels in frames:
        if multi == "avg":
            stacked = torch.stack([model(text) for text in labels], dim=0)
            gt_enc.append(stacked.mean(dim=0))
        elif multi == "single":
            gt_enc.append(model(labels[0]))
        else:
            raise ValueError(f"unsupported --multi {multi!r}")
    return torch.cat(gt_enc, dim=0).detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CLIP-encode per-frame labels. BABEL JSON is not shipped; "
            "pass --annotation_dir pointing at your converted files."
        )
    )
    p.add_argument(
        "--annotation_dir",
        type=str,
        default=str(DEFAULT_ANNOTATION_DIR),
        help=(
            "Directory of {id}.json files with key frame_label "
            "(list of list of strings, 20 fps). Not included in this repo."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for {id}.npy [T, 512] files.",
    )
    p.add_argument(
        "--multi",
        type=str,
        choices=["avg", "single"],
        default="avg",
        help="Several labels on one frame: average CLIP vectors, or keep the first.",
    )
    p.add_argument("--clip_version", type=str, default="ViT-B/32")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)

    if not annotation_dir.is_dir():
        print(
            "BABEL frame-label JSON is not provided in this repository.\n"
            f"Expected a directory of {{id}}.json files at:\n  {annotation_dir}\n\n"
            "Obtain BABEL from https://babel.is.tue.mpg.de/ and convert it so each file is:\n"
            '  { "frame_label": [ ["label", ...], ... ] }\n'
            "with one inner list per 20 fps frame. See prepare/clip_preprocess/README.md."
        )
        sys.exit(1)

    json_files = sorted(p for p in annotation_dir.iterdir() if p.suffix == ".json")
    if not json_files:
        print(f"No .json files in {annotation_dir}")
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available; using CPU")
    else:
        device = torch.device(args.device)

    model = ClipEncoder(args.clip_version, device)
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    n_ok = 0
    n_skip = 0
    for path in json_files:
        out_path = output_dir / f"{path.stem}.npy"
        if out_path.exists():
            n_skip += 1
            continue
        try:
            frames = _load_frame_labels(path)
            arr = _embed_frames(model, frames, args.multi)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  skip {path.name}: {e}")
            continue
        np.save(out_path, arr.astype(np.float32))
        n_ok += 1
        print(f"  {path.name} -> {arr.shape}")

    print(f"Wrote {n_ok} files ({n_skip} already existed) to {output_dir} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
