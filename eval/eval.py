"""Evaluation engine for the 272-dim motion representation.

Follows MotionStreamer's evaluation protocol with their pretrained TMR-based
evaluator (models/Evaluator_272), yielding metrics identical to their paper:

- R@1 / R@2 / R@3 : R-precision (top-1/2/3 retrieval accuracy)
- MM-Dist         : text-motion matching score (Euclidean distance)
- FID             : Frechet Inception Distance on motion embeddings
- Diversity       : mean distance between random motion-embedding pairs

Pipeline per sample:
    ActionPlanSampler -> 16-dim motion latents -> Causal TAE decode ->
    272-dim motion -> Evaluator_272 motion encoder -> metrics vs. GT stats

Reference GT metrics (MotionStreamer paper Table 1, Real motion):
    R@1=70.2, R@2=86.4, R@3=91.4, MM-Dist=15.151, FID=0.002, Div=27.492
"""

import json
import logging
import os
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy import linalg
from tqdm import tqdm

from src.data.dataset_eval_t2m import DATALoader
from src.sampler.actionplan_sampler import ActionPlanSampler
from src.tae.loader import load_tae, load_norm_stats
from models.Evaluator_272.mld.models.architectures.temos.textencoder.distillbert_actor import (
    DistilbertActorAgnosticEncoder,
)
from models.Evaluator_272.mld.models.architectures.temos.motionencoder.actor import (
    ActorAgnosticEncoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_MOTION_LENGTH = 300  # evaluator expects motions padded to 300 frames (30 fps)


# =============================================================================
# Evaluator loading
# =============================================================================

def _load_evaluator_checkpoint(path: str = "models/Evaluator_272/epoch=99.ckpt"):
    """Load the Evaluator_272 checkpoint.

    The checkpoint was pickled with module paths like `mld.data.utils`; in this
    repo the code lives under `models.Evaluator_272.mld.*`, so we temporarily
    alias the old paths in sys.modules while unpickling.
    """
    import sys
    from models.Evaluator_272.mld.data import utils as mld_data_utils
    from models.Evaluator_272.mld import data as mld_data
    from models.Evaluator_272 import mld

    aliases = {"mld": mld, "mld.data": mld_data, "mld.data.utils": mld_data_utils}
    added = [name for name in aliases if name not in sys.modules]
    for name in added:
        sys.modules[name] = aliases[name]
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        for name in added:
            sys.modules.pop(name, None)


def load_evaluator(device) -> Tuple[Any, Any]:
    """Load the TMR-based text and motion encoders (Evaluator_272)."""
    textencoder = DistilbertActorAgnosticEncoder(
        "distilbert-base-uncased", num_layers=4, latent_dim=256
    )
    motionencoder = ActorAgnosticEncoder(
        nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=MAX_MOTION_LENGTH
    )
    ckpt = _load_evaluator_checkpoint()
    state = ckpt["state_dict"]
    textencoder.load_state_dict(
        {k.replace("textencoder.", ""): v for k, v in state.items() if k.startswith("textencoder.")},
        strict=True,
    )
    motionencoder.load_state_dict(
        {k.replace("motionencoder.", ""): v for k, v in state.items() if k.startswith("motionencoder.")},
        strict=True,
    )
    textencoder.eval().to(device)
    motionencoder.eval().to(device)
    return textencoder, motionencoder


# =============================================================================
# Generation helpers
# =============================================================================

def create_sampler(sampler_config: dict, args: SimpleNamespace, device: str) -> ActionPlanSampler:
    """Instantiate an ActionPlanSampler from a sampler config dict.

    Config keys: name (label only), mode ('actionplan' | 'joint' | 'streaming'),
    and optionally steps_per_block (overlap window K).
    """
    kwargs = {k: v for k, v in sampler_config.items() if k != "name"}
    return ActionPlanSampler(
        run_dir=args.run_dir,
        ckpt_path=args.ckpt_path,
        device=device,
        guidance_weight=args.guidance_weight,
        sampling_timesteps=getattr(args, "sampling_timesteps", None),
        **kwargs,
    )


def generate_latents(sampler: ActionPlanSampler, text: str, target_length_30fps: int) -> np.ndarray:
    """Generate 16-dim motion latents for a text prompt at the GT motion length."""
    # The latent model runs at 7.5 fps (30 fps / 4, the TAE downsampling rate).
    target_length_latent = target_length_30fps // 4
    seconds = target_length_latent / 7.5
    result = sampler.sample(text=text, seconds=seconds, fps=7.5)
    return result["features"]


def decode_latents_to_272(
    latents: torch.Tensor,
    tae_model,
    lengths_272: list,
    device: str,
    mean_272: np.ndarray,
    std_272: np.ndarray,
) -> torch.Tensor:
    """Decode motion latents [batch, T, 16] to 272-dim motion [batch, max_len, 272].

    The TAE temporally upsamples by 4x (7.5 fps -> 30 fps) and outputs
    normalized features, which are denormalized with the 272-dim mean/std.
    """
    mean_t = torch.from_numpy(mean_272).float()
    std_t = torch.from_numpy(std_272).float()
    max_length_272 = max(lengths_272)

    decoded_motions = []
    for i in range(latents.shape[0]):
        latent_length = lengths_272[i] // 4
        latent_i = latents[i : i + 1, :latent_length, :].to(device)
        with torch.no_grad():
            decoded = tae_model.forward_decoder(latent_i)  # [1, latent_len*4, 272]
        decoded = decoded.squeeze(0).cpu() * std_t + mean_t
        decoded = decoded[: lengths_272[i]]
        if decoded.shape[0] < max_length_272:
            decoded = torch.cat(
                [decoded, torch.zeros(max_length_272 - decoded.shape[0], 272)], dim=0
            )
        decoded_motions.append(decoded)
    return torch.stack(decoded_motions, dim=0)


def embed_motion_272(
    latents_np: np.ndarray,
    length_30fps: int,
    tae_model,
    motionencoder,
    device,
    eval_mean: np.ndarray,
    eval_std: np.ndarray,
    mean_272: np.ndarray,
    std_272: np.ndarray,
) -> np.ndarray:
    """Decode one latent sequence to 272-dim and embed it with the evaluator."""
    motion_latent = torch.from_numpy(latents_np).float().unsqueeze(0)
    motion_272 = decode_latents_to_272(
        motion_latent, tae_model, [length_30fps], device=str(device),
        mean_272=mean_272, std_272=std_272,
    )
    motion_272_norm = (motion_272.numpy() - eval_mean) / eval_std
    if motion_272_norm.shape[1] < MAX_MOTION_LENGTH:
        padding = np.zeros((1, MAX_MOTION_LENGTH - motion_272_norm.shape[1], 272))
        motion_272_norm = np.concatenate([motion_272_norm, padding], axis=1)
    motion_272_t = torch.from_numpy(motion_272_norm).float().to(device)
    m_length_t = torch.tensor([length_30fps], dtype=torch.long, device=device)
    with torch.no_grad():
        em = motionencoder(motion_272_t, m_length_t).loc
    return em.cpu().numpy().squeeze(0)


def _generation_worker(
    rank: int,
    gpu_id: int,
    indices: list,
    texts: list,
    lengths: list,
    sampler_config: dict,
    args_dict: dict,
    seed: int,
    eval_mean: np.ndarray,
    eval_std: np.ndarray,
    mean_272: np.ndarray,
    std_272: np.ndarray,
    result_path: str,
) -> None:
    """Multi-GPU worker: generate + decode + embed a subset of the test set.

    Each worker loads its own sampler, TAE, and motion encoder on cuda:gpu_id
    (cuda:0 is reserved for the main process). Results go to an npz file.
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    args = SimpleNamespace(**args_dict)
    tae_model = load_tae(device=str(device))
    sampler = create_sampler(sampler_config, args, device=str(device))
    _, motionencoder = load_evaluator(device)

    results = []
    for idx in indices:
        latents = generate_latents(sampler, texts[idx], lengths[idx])
        emb = embed_motion_272(
            latents, lengths[idx], tae_model, motionencoder, device,
            eval_mean, eval_std, mean_272, std_272,
        )
        results.append((idx, emb))

    np.savez(
        result_path,
        indices=np.array([r[0] for r in results]),
        embeddings=np.array([r[1] for r in results]),
    )


# =============================================================================
# Metrics (exact copies of MotionStreamer's implementations)
# =============================================================================

def euclidean_distance_matrix(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    return np.sqrt(np.maximum(d1 + d2 + d3, 0.0))


def calculate_top_k(mat: np.ndarray, top_k: int) -> np.ndarray:
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    return np.concatenate(top_k_list, axis=1)


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    return top_k_mat, matching_score


def calculate_activation_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        logger.warning("FID produced singular product; adding %s to diagonal", eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def calculate_diversity(activation: np.ndarray, diversity_times: int) -> float:
    diversity_times = min(diversity_times, activation.shape[0] // 2)
    first = np.random.choice(activation.shape[0], diversity_times, replace=False)
    second = np.random.choice(activation.shape[0], diversity_times, replace=False)
    return float(linalg.norm(activation[first] - activation[second], axis=1).mean())


def compute_metrics_from_embeddings(
    text_embs: np.ndarray,
    motion_embs: np.ndarray,
    batch_size: int,
    gt_mu: Optional[np.ndarray],
    gt_cov: Optional[np.ndarray],
    diversity_times: int,
) -> Dict[str, float]:
    """R-precision/matching (batch-wise, as in MotionStreamer), diversity, FID."""
    n = motion_embs.shape[0]
    R_precision_accum = np.array([0.0, 0.0, 0.0])
    matching_score_accum = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        temp_R, temp_match = calculate_R_precision(
            text_embs[start:end], motion_embs[start:end], top_k=3, sum_all=True
        )
        R_precision_accum += temp_R
        matching_score_accum += temp_match
    R_precision = R_precision_accum / n

    fid = 0.0
    if gt_mu is not None:
        gen_mu, gen_cov = calculate_activation_statistics(motion_embs)
        fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)

    return {
        "r_precision_top_1": float(R_precision[0]),
        "r_precision_top_2": float(R_precision[1]),
        "r_precision_top_3": float(R_precision[2]),
        "matching_score": float(matching_score_accum / n),
        "fid": float(fid),
        "diversity": calculate_diversity(motion_embs, diversity_times),
        "n_samples": n,
    }


def print_metrics(name: str, metrics: Dict[str, float], ci: Optional[Dict[str, float]] = None):
    if ci is not None:
        print(
            f"{name:24s} | R@1: {metrics['r_precision_top_1']*100:5.2f}±{ci['r_precision_top_1']*100:.2f} | "
            f"R@2: {metrics['r_precision_top_2']*100:5.2f}±{ci['r_precision_top_2']*100:.2f} | "
            f"R@3: {metrics['r_precision_top_3']*100:5.2f}±{ci['r_precision_top_3']*100:.2f} | "
            f"MM-Dist: {metrics['matching_score']:.3f}±{ci['matching_score']:.3f} | "
            f"FID: {metrics['fid']:.3f}±{ci['fid']:.3f} | "
            f"Div: {metrics['diversity']:.3f}±{ci['diversity']:.3f}"
        )
    else:
        print(
            f"{name:24s} | R@1: {metrics['r_precision_top_1']*100:5.2f} | "
            f"R@2: {metrics['r_precision_top_2']*100:5.2f} | "
            f"R@3: {metrics['r_precision_top_3']*100:5.2f} | "
            f"MM-Dist: {metrics['matching_score']:.4f} | "
            f"FID: {metrics['fid']:.4f} | "
            f"Div: {metrics['diversity']:.4f}"
        )


# =============================================================================
# Ground-truth evaluation
# =============================================================================

def run_gt_eval(args: SimpleNamespace) -> Dict[str, Any]:
    """Compute GT metrics with MotionStreamer's exact dataloader and evaluator.

    Returns GT metrics plus the loaded evaluator, dataloader, and GT FID stats
    that the sampler evaluation reuses.
    """
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    val_loader = DATALoader(
        "t2m_272",
        is_test=True,
        batch_size=args.batch_size,
        num_workers=0,
        unit_length=args.unit_length,
        drop_last=True,
        split_file=getattr(args, "split_file", None),
    )
    logger.info("Test dataset size: %d", len(val_loader.dataset))

    textencoder, motionencoder = load_evaluator(device)

    R_precision_accum = np.array([0.0, 0.0, 0.0])
    matching_score_accum = 0.0
    nb_sample = 0
    motion_emb_list = []
    max_samples = getattr(args, "num_samples", None)

    with torch.no_grad():
        for batch_idx, (text, pose, m_length) in enumerate(val_loader):
            pose = pose.to(device).float()
            et = textencoder(text).loc
            em = motionencoder(pose, m_length).loc
            motion_emb_list.append(em.cpu())

            temp_R, temp_match = calculate_R_precision(
                et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
            )
            R_precision_accum += temp_R
            matching_score_accum += temp_match
            nb_sample += pose.shape[0]

            if batch_idx % 20 == 0:
                logger.info(
                    "  GT batch %d/%d: R@1 = %.2f%%",
                    batch_idx, len(val_loader), R_precision_accum[0] / nb_sample * 100,
                )
            if max_samples is not None and nb_sample >= max_samples:
                break

    motion_emb_np = torch.cat(motion_emb_list, dim=0).numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_emb_np)
    gt_metrics = {
        "r_precision_top_1": float(R_precision_accum[0] / nb_sample),
        "r_precision_top_2": float(R_precision_accum[1] / nb_sample),
        "r_precision_top_3": float(R_precision_accum[2] / nb_sample),
        "matching_score": float(matching_score_accum / nb_sample),
        "fid": 0.0,  # by definition for GT
        "diversity": calculate_diversity(motion_emb_np, args.diversity_times),
        "n_samples": nb_sample,
    }

    print("\nGround Truth metrics (MotionStreamer protocol):")
    print_metrics("GT (Real)", gt_metrics)

    return {
        "gt_metrics": gt_metrics,
        "gt_mu": gt_mu,
        "gt_cov": gt_cov,
        "textencoder": textencoder,
        "motionencoder": motionencoder,
        "val_loader": val_loader,
        "device": device,
    }


# =============================================================================
# Sampler evaluation
# =============================================================================

def effective_num_gpus(args: SimpleNamespace) -> int:
    """Number of generation workers. cuda:0 is reserved for the evaluator, so
    multi-GPU generation needs at least 2 visible GPUs."""
    requested = getattr(args, "num_gpus", 1)
    if requested <= 1 or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return 1
    return min(requested, torch.cuda.device_count() - 1)


def run_single_replication(
    args: SimpleNamespace,
    gt_results: Dict[str, Any],
    sampler: Optional[ActionPlanSampler],
    sampler_config: dict,
    tae_model,
    mean_272: np.ndarray,
    std_272: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    """Generate motions for the whole test set once and compute metrics."""
    device = gt_results["device"]
    val_loader = gt_results["val_loader"]
    textencoder = gt_results["textencoder"]
    motionencoder = gt_results["motionencoder"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    eval_mean = val_loader.dataset.mean
    eval_std = val_loader.dataset.std
    max_samples = getattr(args, "num_samples", None)

    # Collect (text, length) pairs in dataloader order (random text choice +
    # unit-length cropping happen inside the dataset, matching the GT protocol).
    all_texts, all_lengths = [], []
    for text, _, m_length in val_loader:
        all_texts.extend(text)
        all_lengths.extend(int(l) for l in m_length)
        if max_samples is not None and len(all_texts) >= max_samples:
            break
    n_samples = len(all_texts)

    num_gpus = min(effective_num_gpus(args), n_samples)

    if num_gpus > 1:
        indices_per_rank = [list(range(r, n_samples, num_gpus)) for r in range(num_gpus)]
        base = getattr(args, "output_dir", None) or args.run_dir
        tmp_dir = os.path.join(base, ".eval_multigpu_tmp", f"{os.getpid()}_{uuid.uuid4().hex[:8]}")
        os.makedirs(tmp_dir, exist_ok=True)
        result_paths = [os.path.join(tmp_dir, f"worker_{r}.npz") for r in range(num_gpus)]

        args_dict = {
            "run_dir": args.run_dir,
            "ckpt_path": args.ckpt_path,
            "guidance_weight": args.guidance_weight,
            "sampling_timesteps": getattr(args, "sampling_timesteps", None),
        }
        ctx = mp.get_context("spawn")
        processes = []
        for rank in range(num_gpus):
            p = ctx.Process(
                target=_generation_worker,
                args=(
                    rank, rank + 1, indices_per_rank[rank], all_texts, all_lengths,
                    sampler_config, args_dict, seed,
                    eval_mean, eval_std, mean_272, std_272, result_paths[rank],
                ),
            )
            p.start()
            processes.append((rank, p))
        for rank, p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"Worker {rank} (cuda:{rank + 1}) exited with code {p.exitcode}. "
                    "Likely OOM or CUDA error; try --num_gpus 1."
                )

        results_by_idx = {}
        for path in result_paths:
            with np.load(path) as data:
                for idx, emb in zip(data["indices"], data["embeddings"]):
                    results_by_idx[int(idx)] = emb
        shutil.rmtree(tmp_dir, ignore_errors=True)
        motion_embs = np.stack([results_by_idx[i] for i in range(n_samples)], axis=0)
    else:
        if sampler is None:
            sampler = create_sampler(sampler_config, args, str(device))
        motion_emb_list = []
        for i in tqdm(range(n_samples), desc=f"Generating [{sampler_config['name']}] (seed={seed})"):
            latents = generate_latents(sampler, all_texts[i], all_lengths[i])
            motion_emb_list.append(
                embed_motion_272(
                    latents, all_lengths[i], tae_model, motionencoder, device,
                    eval_mean, eval_std, mean_272, std_272,
                )
            )
        motion_embs = np.stack(motion_emb_list, axis=0)

    # Text embeddings on the main process
    text_emb_list = []
    with torch.no_grad():
        for start in range(0, n_samples, args.batch_size):
            et = textencoder(all_texts[start : start + args.batch_size]).loc
            text_emb_list.append(et.cpu().numpy())
    text_embs = np.concatenate(text_emb_list, axis=0)

    return compute_metrics_from_embeddings(
        text_embs, motion_embs, args.batch_size,
        gt_results["gt_mu"], gt_results["gt_cov"], args.diversity_times,
    )


def run_model_eval(args: SimpleNamespace, gt_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate every sampler in args.samplers, with replications for CIs."""
    device = gt_results["device"]
    replication_times = getattr(args, "replication_times", 1)

    tae_model = load_tae(device=str(device))
    mean_272, std_272 = load_norm_stats()

    all_sampler_results = {}
    for sampler_config in args.samplers:
        sampler_name = sampler_config["name"]
        print(f"\n=== Evaluating sampler: {sampler_name} (config: {sampler_config}, "
              f"guidance={args.guidance_weight}) ===")

        # In multi-GPU mode each worker builds its own sampler.
        sampler = None if effective_num_gpus(args) > 1 else create_sampler(sampler_config, args, str(device))

        metrics_per_rep: Dict[str, list] = {
            k: [] for k in [
                "r_precision_top_1", "r_precision_top_2", "r_precision_top_3",
                "matching_score", "fid", "diversity",
            ]
        }
        for rep in range(replication_times):
            if replication_times > 1:
                print(f"--- Replication {rep + 1}/{replication_times} ---")
            metrics = run_single_replication(
                args, gt_results, sampler, sampler_config,
                tae_model, mean_272, std_272, seed=args.seed + rep,
            )
            for k in metrics_per_rep:
                metrics_per_rep[k].append(metrics[k])

        final_metrics, final_ci = {}, {}
        for k, values in metrics_per_rep.items():
            arr = np.array(values)
            final_metrics[k] = float(arr.mean())
            final_ci[k] = float(1.96 * arr.std() / np.sqrt(replication_times)) if replication_times > 1 else 0.0
        final_metrics["n_samples"] = metrics["n_samples"]
        final_metrics["replication_times"] = replication_times

        print(f"\n--- {sampler_name} results ---")
        print_metrics(sampler_name, final_metrics, ci=final_ci if replication_times > 1 else None)

        all_sampler_results[sampler_name] = {
            "metrics": final_metrics,
            "metrics_ci": final_ci,
            "metrics_per_replication": metrics_per_rep,
            "config": sampler_config,
        }
        save_sampler_results(
            args, sampler_name, all_sampler_results[sampler_name], gt_results["gt_metrics"]
        )

    print("\n=== Summary ===")
    print_metrics("GT (Real)", gt_results["gt_metrics"])
    for sampler_name, result in all_sampler_results.items():
        print_metrics(sampler_name, result["metrics"])

    return all_sampler_results


def save_sampler_results(
    args: SimpleNamespace,
    sampler_name: str,
    sampler_result: Dict[str, Any],
    gt_metrics: Dict[str, Any],
) -> Path:
    """Write per-sampler metrics JSON to <output_dir or run_dir/eval_results>/<name>/."""
    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir) / sampler_name
    else:
        output_dir = Path(args.run_dir) / "eval_results" / sampler_name
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_to_save = {
        "timestamp": timestamp,
        "sampler_name": sampler_name,
        "sampler_config": sampler_result["config"],
        "metrics": sampler_result["metrics"],
        "metrics_ci": sampler_result["metrics_ci"],
        "metrics_per_replication": sampler_result["metrics_per_replication"],
        "gt_metrics": gt_metrics,
        "evaluation_config": {
            "run_dir": str(args.run_dir),
            "ckpt_path": str(args.ckpt_path),
            "guidance_weight": args.guidance_weight,
            "sampling_timesteps": getattr(args, "sampling_timesteps", None),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "unit_length": args.unit_length,
            "replication_times": args.replication_times,
            "diversity_times": args.diversity_times,
            "num_samples": getattr(args, "num_samples", None),
            "split_file": getattr(args, "split_file", None),
        },
    }
    output_file = output_dir / f"eval_{timestamp}_seed{args.seed}.json"
    for path in (output_file, output_dir / "latest.json"):
        with open(path, "w") as f:
            json.dump(results_to_save, f, indent=2)
    logger.info("Saved results for %s to: %s", sampler_name, output_file)
    return output_file


def run_eval(args: SimpleNamespace) -> Dict[str, Any]:
    """Run GT eval followed by sampler eval; see eval.py for the CLI."""
    gt_results = run_gt_eval(args)
    sampler_metrics = run_model_eval(args, gt_results)
    return {
        "gt_metrics": gt_results["gt_metrics"],
        "sampler_metrics": sampler_metrics,
    }
