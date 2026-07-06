#!/usr/bin/env python3
"""
Download 272-dim MotionStreamer data and compute 16-dim TAE latents.

Reference: https://github.com/Li-xingXiao/272-dim-Motion-Representation

This script:
1. Downloads the preprocessed 272-dim HumanML3D dataset from HuggingFace
2. Optionally computes 16-dim TAE latents from 272-dim motion

Provide your own annotations, stats, and text stats.

Usage:
    python prepare/download_streamer272_data.py

    # Skip latent computation (272-dim only)
    python prepare/download_streamer272_data.py --skip_latents

    # Only compute latents (272-dim data must already exist)
    python prepare/download_streamer272_data.py --only_latents --skip_download

    # Reproducible latents (same as MotionStreamer get_latent.py window logic, no reference token)
    python prepare/download_streamer272_data.py --seed 42

Requirements:
    - huggingface_hub: pip install huggingface_hub
    - torch: for TAE encoding (when computing latents)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_from_huggingface(local_dir: str, dataset_name: str = "lxxiao/272-dim-HumanML3D") -> bool:
    """
    Download dataset from HuggingFace using huggingface_hub.
    
    Args:
        local_dir: Local directory to save the dataset
        dataset_name: HuggingFace dataset repository name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"Downloading {dataset_name} to {local_dir}...")
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=local_dir,
            resume_download=True,
        )
        print("Download complete!")
        return True
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        print("\nAlternatively, download manually with:")
        print(f"  huggingface-cli download --repo-type dataset {dataset_name} --local-dir {local_dir}")
        return False
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def unzip_data(data_dir: str):
    """Unzip texts.zip and motion_data.zip if they exist."""
    import zipfile
    
    for zip_name in ["texts.zip", "motion_data.zip"]:
        zip_path = os.path.join(data_dir, zip_name)
        if os.path.exists(zip_path):
            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted {zip_name}")


def compute_latents(
    motion_dir: str,
    latent_dir: str,
    tae_checkpoint: Optional[str] = None,
    norm_stats_dir: Optional[str] = None,
    device: str = None,
    seed: Optional[int] = None,
) -> bool:
    """
    Compute 16-dim TAE latents from 272-dim motion files.
    
    Matches MotionStreamer get_latent.py behavior (without reference token):
    - unit_length=4: truncate motion length to multiple of 4
    - When length not divisible by 4: random start index (like dataset_tae_tokenizer)
    - min_motion_len=40: skip motions shorter than 40 frames
    
    Each latent file has shape (n_latent_frames, 16).
    
    Args:
        motion_dir: Directory containing motion_data/*.npy (272-dim)
        latent_dir: Output directory for latent/*.npy (16-dim)
        tae_checkpoint: Path to TAE checkpoint. If None, uses default.
        norm_stats_dir: Path to Mean.npy/Std.npy. If None, uses motion_dir/../mean_std
        device: cuda, mps, or cpu. Auto-detected if None.
        seed: Random seed for window sampling. If set, runs are reproducible.
    
    Returns:
        True if successful
    """
    import random
    import numpy as np
    import torch
    from tqdm import tqdm
    
    if seed is not None:
        random.seed(seed)
    
    motion_dir = Path(motion_dir)
    latent_dir = Path(latent_dir)
    
    unit_length = 4
    min_motion_len = 40
    
    if not motion_dir.exists():
        print(f"Error: Motion directory not found: {motion_dir}")
        return False
    
    # Get motion IDs from motion_data
    motion_files = sorted([f.stem for f in motion_dir.glob("*.npy")])
    if not motion_files:
        print(f"Error: No .npy files found in {motion_dir}")
        return False
    
    # Device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    
    # Load TAE
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from src.tae import load_tae, encode_motion
    except ImportError as e:
        print(f"Error: Could not import TAE module: {e}")
        return False
    
    tae_checkpoint = tae_checkpoint or str(Path(__file__).parent.parent / "models" / "Causal_TAE" / "net_last.pth")
    if not Path(tae_checkpoint).exists():
        print(f"Error: TAE checkpoint not found: {tae_checkpoint}")
        return False
    
    norm_stats_dir = norm_stats_dir or str(motion_dir.parent / "mean_std")
    
    print(f"Loading TAE from {tae_checkpoint}...")
    model = load_tae(checkpoint_path=tae_checkpoint, device=device)
    
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Computing latents for {len(motion_files)} motions...")
    failed = 0
    for motion_id in tqdm(motion_files, desc="Encoding"):
        motion_path = motion_dir / f"{motion_id}.npy"
        try:
            motion = np.load(motion_path)
        except Exception as e:
            print(f"Warning: Could not load {motion_id}: {e}")
            failed += 1
            continue
        
        if motion.shape[-1] != 272:
            print(f"Warning: {motion_id} has wrong shape {motion.shape}, expected (T, 272)")
            failed += 1
            continue
        
        # Match dataset_tae_tokenizer: skip short motions, apply random window when len % 4 != 0
        if len(motion) < min_motion_len:
            failed += 1
            continue
        m_length = (len(motion) // unit_length) * unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]
        
        # Encode (encode_motion expects (seq, 272) or (batch, seq, 272))
        try:
            latent = encode_motion(
                motion,
                model=model,
                device=device,
                normalize=True,
                norm_stats_dir=norm_stats_dir,
            )
        except Exception as e:
            print(f"Warning: Could not encode {motion_id}: {e}")
            failed += 1
            continue
        
        latent = latent.numpy() if isinstance(latent, torch.Tensor) else np.asarray(latent)
        np.save(latent_dir / f"{motion_id}.npy", latent.astype(np.float32))
    
    if failed > 0:
        print(f"Warning: {failed} motions failed to encode")
    print(f"Saved {len(motion_files) - failed} latent files to {latent_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and prepare 272-dim MotionStreamer data")
    parser.add_argument("--data_dir", type=str, default="datasets/motions/humanml3d_272",
                        help="Directory to download/store the dataset")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download (use if data already exists)")
    parser.add_argument("--skip_latents", action="store_true",
                        help="Skip latent computation (272-dim only)")
    parser.add_argument("--only_latents", action="store_true",
                        help="Only compute latents from existing 272-dim data (skip download)")
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Output directory for 16-dim latents (default: datasets/motions/t2m_latents)")
    parser.add_argument("--tae_checkpoint", type=str, default=None,
                        help="Path to TAE checkpoint (default: models/Causal_TAE/net_last.pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for encoding (cuda, mps, cpu). Auto-detected if not specified.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for window sampling (when motion length not divisible by 4). "
                             "If set, runs are reproducible.")
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("=" * 60)
    print("272-dim MotionStreamer Dataset Preparation")
    print("=" * 60)
    print()
    
    # Fast path: only compute latents from existing 272-dim data
    if args.only_latents:
        motion_dir = os.path.join(args.data_dir, "motion_data")
        latent_dir = args.latent_dir or os.path.join(Path(__file__).parent.parent, "datasets", "motions", "t2m_latents")
        norm_stats_dir = os.path.join(args.data_dir, "mean_std")
        print("[Only Latents Mode] Computing 16-dim TAE latents from 272-dim motion...")
        if not compute_latents(
            motion_dir=motion_dir,
            latent_dir=latent_dir,
            tae_checkpoint=args.tae_checkpoint,
            norm_stats_dir=norm_stats_dir,
            device=args.device,
            seed=args.seed,
        ):
            return 1
        print()
        print("=" * 60)
        print("Latents computed!")
        print("=" * 60)
        print()
        print("You can now train the ActionPlan model:")
        print("  python train.py --config-name=train_actionplan")
        print()
        return 0
    
    # Step 1: Download
    if not args.skip_download:
        print("[Step 1/4] Downloading dataset from HuggingFace...")
        if not download_from_huggingface(args.data_dir):
            print("Download failed. You can try downloading manually and re-run with --skip_download")
            return 1
    else:
        print("[Step 1/3] Skipping download (--skip_download)")
    print()
    
    # Step 2: Unzip
    print("[Step 2/3] Extracting archives...")
    unzip_data(args.data_dir)
    print()
    
    # Step 3: Compute latents (272-dim -> 16-dim for latent training)
    if not args.skip_latents:
        print("[Step 3/3] Computing 16-dim TAE latents...")
        motion_dir = os.path.join(args.data_dir, "motion_data")
        latent_dir = args.latent_dir or os.path.join(project_root, "datasets", "motions", "t2m_latents")
        norm_stats_dir = os.path.join(args.data_dir, "mean_std")
        if not compute_latents(
            motion_dir=motion_dir,
            latent_dir=latent_dir,
            tae_checkpoint=args.tae_checkpoint,
            norm_stats_dir=norm_stats_dir,
            device=args.device,
            seed=args.seed,
        ):
            print("Latent computation failed. You can re-run with --skip_latents to skip.")
        print()
    else:
        print("[Step 3/3] Skipping latent computation (--skip_latents)")
        print()
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

