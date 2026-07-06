"""Benchmark streaming latency (paper runtime table).

Measures, over random test-set prompts, the time until each motion latent
(= 4 frames = one "token") is fully denoised in streaming mode: the action
plan is denoised together with the first motion latent, then the remaining
latents follow in raster order with 2 denoising steps per newly added latent.

Reported numbers:
- First latent latency  ("First" column):  time until latent 0 is ready
- Subsequent latency    ("Others" column): time from latent i-1 ready to latent i ready
- Decode timings for the Causal TAE (full sequence and per-latent serving)

Paper (single NVIDIA A100): First 146 ms, Others 40 ms.

Usage:
    python benchmark_latency.py
"""

import argparse
import json
import os
import random
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from src.data.text_motion import load_annotations, read_split
from src.model.actionplan_rectified_flow import MOTION_DIM
from src.model.utils import masked
from src.sampler.actionplan_sampler import ActionPlanSampler
from src.tae.loader import load_tae, decode_latents, load_norm_stats

ANNOTATION_DATASET = "humanml3d_actionplan_merged"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def sample_random_prompts(split: str, k: int, seed: int) -> List[Tuple[str, float]]:
    """Sample k random (text, duration_seconds) pairs from the annotations."""
    annotation_path = os.path.join("datasets", "annotations", ANNOTATION_DATASET)
    keyids = read_split(annotation_path, split)
    annotations = load_annotations(annotation_path)
    keyids = [kid for kid in keyids if kid in annotations]

    rng = random.Random(seed)
    chosen = rng.sample(keyids, min(k, len(keyids)))

    result = []
    for keyid in chosen:
        ann = annotations[keyid]["annotations"][0]
        seconds = float(ann.get("end", 10.0) - ann.get("start", 0.0))
        result.append((str(ann["text"]), seconds if seconds > 0 else 5.0))
    return result


@torch.no_grad()
def run_streaming_with_timing(
    sampler: ActionPlanSampler,
    text: str,
    duration: int,
    effective_length: int,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """Run the streaming schedule and record when each latent becomes clean.

    Mirrors ActionPlanSampler's streaming mode (phase 1: text + first latent
    together; phase 2: raster-order pyramid) but with timers around each step.
    Returns (motion_latents [T, 16], time_to_denoise, subsequent_latency), in seconds.
    """
    device = sampler.device
    motion_steps = sampler.diffusion.motion_steps

    infos = {
        "all_lengths": [effective_length],
        "all_texts": [text],
        "featsname": sampler.featsname,
        "guidance_weight": float(sampler.guidance_weight),
        "stochastic_sampling": sampler.stochastic_sampling,
        "variance_alpha": sampler.variance_alpha,
        "sampling_temperature": sampler.sampling_temperature,
    }
    tx = sampler._build_text_embeddings([text])
    tx_uncond = sampler._build_text_embeddings([""])
    mask = torch.zeros((1, duration), device=device, dtype=torch.bool)
    mask[:, :effective_length] = True
    y = {
        "length": [effective_length],
        "mask": mask,
        "tx": sampler.diffusion.prepare_tx_emb(tx),
        "tx_uncond": sampler.diffusion.prepare_tx_emb(tx_uncond),
        "infos": infos,
    }

    xt = torch.randn((1, duration, sampler.diffusion.denoiser.nfeats), device=device)
    time_to_denoise = [float("inf")] * effective_length
    t_start = time.perf_counter()

    # Phase 1: action plan + first latents together (pyramid ramp-up)
    xt = sampler._phase1_denoise_text_and_first_motion(
        xt, y, effective_length, progress_bar=None, use_ema=True
    )
    _sync(device)
    phase1_time = time.perf_counter() - t_start

    target_levels = sampler._phase1_pyramid_target_levels(effective_length, motion_steps)
    for i in range(effective_length):
        if target_levels[i] == 0:
            time_to_denoise[i] = phase1_time
    level_state = torch.tensor(
        target_levels + [0] * (duration - effective_length),
        dtype=torch.long, device=device,
    )
    remaining_frames = [i for i in range(effective_length) if target_levels[i] > 0]
    remaining_set = set(remaining_frames)
    tau_grid = torch.linspace(1.0, 0.0, motion_steps + 1, device=device, dtype=xt.dtype)

    # Phase 2: raster-order pyramid over remaining latents
    picked: List[int] = []
    frame_order = list(remaining_frames)
    frame_order_idx = 0
    while remaining_frames and level_state[remaining_frames].max().item() > 0:
        if len(picked) < len(remaining_frames):
            while frame_order_idx < len(frame_order):
                new_idx = frame_order[frame_order_idx]
                frame_order_idx += 1
                if new_idx in remaining_set and new_idx not in picked and level_state[new_idx] > 0:
                    picked.append(new_idx)
                    break
        for _ in range(sampler.steps_per_block):
            if level_state[remaining_frames].max().item() <= 0:
                break
            xt = sampler._phase2_step_frames(
                xt, y, level_state, picked, duration, effective_length,
                tau_grid, use_ema=True,
            )
            _sync(device)
            now = time.perf_counter() - t_start
            for i in range(effective_length):
                if level_state[i].item() == 0 and time_to_denoise[i] == float("inf"):
                    time_to_denoise[i] = now

    total = time.perf_counter() - t_start
    time_to_denoise = [t if t != float("inf") else total for t in time_to_denoise]
    subsequent_latency = [
        time_to_denoise[i] - time_to_denoise[i - 1] if i > 0 else time_to_denoise[0]
        for i in range(effective_length)
    ]

    x_out = sampler.diffusion.motion_normalizer.inverse(masked(xt, mask))
    latents_np = x_out[0, :effective_length, :MOTION_DIM].detach().cpu().numpy()
    return latents_np, time_to_denoise, subsequent_latency


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ActionPlan streaming latency (paper runtime table).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_dir", type=str, default="outputs/actionplan", help="Model run directory")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint (default: latest in run_dir)")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of random prompts to benchmark")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before measurement")
    parser.add_argument("--split", type=str, default="test", help="Dataset split for prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt sampling")
    parser.add_argument("--steps_per_block", type=int, default=2, help="Denoising steps per newly added latent (paper: 2)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    args = parser.parse_args()

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )

    sampler = ActionPlanSampler(
        run_dir=args.run_dir,
        ckpt_path=args.ckpt_path,
        device=str(device),
        mode="streaming",
        steps_per_block=args.steps_per_block,
    )
    prompts = sample_random_prompts(args.split, args.num_runs + args.warmup, seed=args.seed)

    for i in range(args.warmup):
        text, seconds = prompts[i]
        dur, eff = sampler._resolve_duration(max(1, int(round(sampler.fps * seconds))))
        run_streaming_with_timing(sampler, text, dur, eff)
        print(f"Warmup {i + 1}/{args.warmup} done")

    tae_model = load_tae(device=device)
    mean_272, std_272 = load_norm_stats()

    first_latencies: List[float] = []
    subsequent_latencies: List[float] = []
    decode_full_times: List[float] = []
    per_latent_decode_times: List[float] = []

    for i in range(args.warmup, args.warmup + args.num_runs):
        text, seconds = prompts[i]
        dur, eff = sampler._resolve_duration(max(1, int(round(sampler.fps * seconds))))
        latents_np, time_to_denoise, subsequent = run_streaming_with_timing(sampler, text, dur, eff)

        first_latencies.append(time_to_denoise[0] * 1000)
        subsequent_latencies.extend(s * 1000 for s in subsequent[1:])

        # Full-sequence decode with the Causal TAE
        _sync(device)
        t0 = time.perf_counter()
        decode_latents(latents_np, model=tae_model, device=device,
                       remove_reference_token=False, denormalize=True)
        _sync(device)
        decode_full_times.append((time.perf_counter() - t0) * 1000)

        # Per-latent decode: time to decode & serve each new latent's prefix
        for j in range(latents_np.shape[0]):
            prefix = latents_np[np.newaxis, : j + 1]
            _sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                decoded = tae_model.forward_decoder(torch.from_numpy(prefix).float().to(device)).cpu()
                decoded = decoded * torch.from_numpy(std_272).float() + torch.from_numpy(mean_272).float()
            _sync(device)
            per_latent_decode_times.append((time.perf_counter() - t0) * 1000)

        if (i - args.warmup + 1) % 10 == 0:
            print(f"Run {i - args.warmup + 1}/{args.num_runs} done")

    def stats(arr: List[float]) -> Tuple[float, float]:
        a = np.asarray(arr, dtype=np.float64)
        return float(a.mean()), float(a.std())

    m_first, s_first = stats(first_latencies)
    m_next, s_next = stats(subsequent_latencies)
    m_full, s_full = stats(decode_full_times)
    m_serve, s_serve = stats(per_latent_decode_times)

    print("\n" + "=" * 70)
    print("ActionPlan Streaming Latency Benchmark")
    print("=" * 70)
    print(f"Config: {args.num_runs} runs, steps_per_block={sampler.steps_per_block}, split={args.split}")
    print("-" * 70)
    print(f"First latent latency:       {m_first:.2f} ± {s_first:.2f} ms")
    print(f"Subsequent latent latency:  {m_next:.2f} ± {s_next:.2f} ms")
    print(f"Decode full (Causal TAE):   {m_full:.2f} ± {s_full:.2f} ms")
    print(f"Per-latent decode (serve):  {m_serve:.2f} ± {s_serve:.2f} ms")
    print("=" * 70)

    results = {
        "config": {
            "run_dir": os.path.abspath(args.run_dir),
            "num_runs": args.num_runs,
            "steps_per_block": int(sampler.steps_per_block),
            "split": args.split,
            "seed": args.seed,
        },
        "results": {
            "first_latent_latency_ms": {"mean": m_first, "std": s_first},
            "subsequent_latent_latency_ms": {"mean": m_next, "std": s_next},
            "decode_full_ms": {"mean": m_full, "std": s_full},
            "per_latent_decode_serve_ms": {"mean": m_serve, "std": s_serve},
        },
    }
    out_path = os.path.join(args.run_dir, "benchmark_streaming_latency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
