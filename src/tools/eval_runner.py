import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _get_run_dir(trainer) -> Optional[str]:
    loggers = []
    if getattr(trainer, "loggers", None):
        loggers = list(trainer.loggers)
    elif getattr(trainer, "logger", None) is not None:
        loggers = [trainer.logger]

    for lg in loggers:
        save_dir = getattr(lg, "save_dir", None)
        if save_dir:
            return str(save_dir)
        log_dir = getattr(lg, "log_dir", None)
        if log_dir:
            return str(Path(log_dir).parent)

    default_root = getattr(trainer, "default_root_dir", None)
    return str(default_root) if default_root else None


def _get_latest_checkpoint(trainer, run_dir: Optional[str]) -> Optional[str]:
    callbacks = getattr(trainer, "checkpoint_callbacks", []) or []
    for cb in callbacks:
        last_path = getattr(cb, "last_model_path", None)
        if last_path and os.path.isfile(last_path):
            return last_path

    for cb in callbacks:
        best_path = getattr(cb, "best_model_path", None)
        if best_path and os.path.isfile(best_path):
            return best_path

    if run_dir:
        ckpt_dir = os.path.join(run_dir, "logs", "checkpoints")
        if os.path.isdir(ckpt_dir):
            candidates = []
            for name in os.listdir(ckpt_dir):
                if name.endswith(".ckpt"):
                    path = os.path.join(ckpt_dir, name)
                    try:
                        mtime = os.path.getmtime(path)
                    except Exception:
                        mtime = 0.0
                    candidates.append((mtime, path))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]
    return None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_wandb_logger(trainer):
    try:
        from pytorch_lightning.loggers import WandbLogger
    except Exception:
        return None

    loggers = []
    if getattr(trainer, "loggers", None):
        loggers = list(trainer.loggers)
    elif getattr(trainer, "logger", None) is not None:
        loggers = [trainer.logger]

    for lg in loggers:
        if isinstance(lg, WandbLogger):
            return lg
    return None


def _log_eval_metrics(trainer, prefix: str, metrics: dict, step: int, epoch: int) -> None:
    wandb_logger = _get_wandb_logger(trainer)
    if wandb_logger is None:
        return

    run = getattr(wandb_logger, "experiment", None)
    if run is None or not hasattr(run, "log"):
        return

    payload = {
        f"{prefix}/epoch": epoch,
        f"{prefix}/step": step,
    }
    for key, value in metrics.items():
        try:
            payload[f"{prefix}/{key}"] = float(value)
        except Exception:
            continue

    run.log(payload, step=step)


def infer_is_latent_model(pl_module) -> bool:
    motion_normalizer = getattr(pl_module, "motion_normalizer", None)
    base_dir = getattr(motion_normalizer, "base_dir", "") if motion_normalizer is not None else ""
    if isinstance(base_dir, str) and ("latents" in base_dir in base_dir):
        return True

    denoiser = getattr(pl_module, "denoiser", None)
    nfeats = getattr(denoiser, "nfeats", None)
    if isinstance(nfeats, int) and nfeats <= 32:
        return True

    return False


def run_validation_eval(trainer, pl_module, is_latent: bool) -> None:
    if trainer.sanity_checking:
        return
    if not trainer.is_global_zero:
        return
    if getattr(trainer, "fast_dev_run", False):
        return

    run_dir = _get_run_dir(trainer)
    ckpt_path = _get_latest_checkpoint(trainer, run_dir)
    if not run_dir:
        logger.warning("Validation eval skipped: could not determine run_dir.")
        return
    if not ckpt_path:
        logger.warning("Validation eval skipped: checkpoint not found yet.")
        return

    actionplan_root = Path(__file__).resolve().parents[2]
    project_root = actionplan_root.parent
    device = _get_device()
    seed = int(getattr(trainer, "current_epoch", 0))
    guidance_weight = float(getattr(pl_module, "guidance_weight", 3.0))

    if is_latent:
        from eval.eval import run_eval

        humanml3d_272_dir = actionplan_root / "datasets" / "motions" / "humanml3d_272"
        evaluator_ckpt = actionplan_root / "models" / "Evaluator_272" / "epoch=99.ckpt"
        split_file = "val.txt"
        split_path = humanml3d_272_dir / "split" / split_file

        if not humanml3d_272_dir.exists():
            logger.warning("Validation eval skipped: %s not found.", humanml3d_272_dir)
            return
        if not evaluator_ckpt.exists():
            logger.warning("Validation eval skipped: %s not found.", evaluator_ckpt)
            return
        if not split_path.exists():
            logger.warning("Validation eval skipped: split file not found at %s", split_path)
            return

        sampler_config = {
            "name": "offline",
            "mode": "actionplan",
            "steps_per_block": 2,
        }

        args = SimpleNamespace(
            humanml3d_272_dir=str(humanml3d_272_dir),
            evaluator_ckpt=str(evaluator_ckpt),
            num_samples=None,
            batch_size=32,
            device=device,
            seed=seed,
            apply_cropping=True,
            unit_length=4,
            replication_times=1,
            diversity_times=300,
            run_dir=str(run_dir),
            ckpt_path=str(ckpt_path),
            guidance_weight=guidance_weight,
            output_dir=None,
            split_file=split_file,
            samplers=[sampler_config],
        )

        logger.info(
            "Running latent validation eval (actionplan_pyramid_random s2) on %s",
            split_file,
        )
        results = run_eval(args)
        if isinstance(results, dict):
            sampler_metrics = results.get("sampler_metrics", {})
            for sampler_name, sampler_result in sampler_metrics.items():
                metrics = sampler_result.get("metrics", {})
                _log_eval_metrics(
                    trainer,
                    f"val_eval/{sampler_name}",
                    metrics,
                    step=int(getattr(trainer, "global_step", 0)),
                    epoch=int(getattr(trainer, "current_epoch", 0)),
                )
        return

    # 205-dim / non-latent validation not supported in trimmed codebase
    logger.warning(
        "Validation eval skipped: non-latent (205-dim) models not supported. "
        "Only latent (272-dim) models are supported."
    )
