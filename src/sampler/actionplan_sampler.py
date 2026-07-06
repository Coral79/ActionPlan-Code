"""ActionPlan sampler for rectified flow.

Two modes:
  actionplan: phase1 (full text denoise) + phase2 (random pyramid, 2 steps)
  streaming:  phase1 (text+first motion together) + phase2 (pyramid for rest, with on_frame_ready)

Respects inference_length (e.g. 78): latent sequences are padded to that length;
only frames 0..effective_length-1 are treated as motion content.
"""

import os
import queue
import re
import threading
import logging
from typing import Optional, Dict, Any, List, Tuple, Set, Union, Callable, Generator

import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from src.config import read_config
from src.model.text_encoder import TextToEmb
from src.model.actionplan_rectified_flow import ActionPlanRectifiedFlow, MOTION_DIM, TEXT_DIM
from src.model.utils import masked


logger = logging.getLogger(__name__)

TEXT_SLICE = slice(MOTION_DIM, MOTION_DIM + TEXT_DIM)
MOTION_SLICE = slice(0, MOTION_DIM)
FULL_SLICE = slice(0, MOTION_DIM + TEXT_DIM)



def _ensure_headless_gl() -> None:
    """Set env vars for headless OpenGL (EGL) when no display."""
    if os.environ.get("DISPLAY") in (None, ""):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")


# -----------------------------------------------------------------------------
# ActionPlanSampler
# -----------------------------------------------------------------------------


class ActionPlanSampler:
    """Dual-stream sampler for ActionPlan rectified flow.

    Modes:
      actionplan: phase1 (text only) + phase2 (random pyramid)
      joint: phase1 (text only) + phase2 (all motion latents denoised in parallel)
      streaming: phase1 (text+first motion) + phase2 (frame-wise pyramid with on_frame_ready)
      streaming_block: phase1 (text+first block) + phase2 (block-wise pyramid with on_block_ready)
    """

    MODES = ("actionplan", "joint", "streaming", "streaming_block")

    def __init__(
        self,
        run_dir: str,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        guidance_weight: float = 3.0,
        abs_root: bool = False,
        mode: str = "actionplan",
        remaining_strategy: Optional[str] = None,
        num_blocks: int = 10,
        steps_per_block: int = 2,
        pick_lowest_variance: bool = True,
        sampling_timesteps: Optional[int] = None,
    ) -> None:
        """Load diffusion model, text encoder, and config from run_dir."""
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _ensure_headless_gl()

        if remaining_strategy is not None:
            if remaining_strategy == "pyramid_random":
                mode = "actionplan"
            elif remaining_strategy in ("streaming_block", "joint"):
                mode = remaining_strategy
            else:
                mode = "streaming"
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError(
                f"mode must be one of {self.MODES}, got {self.mode!r}"
            )

        self.run_dir = os.path.abspath(run_dir)
        self.guidance_weight = float(guidance_weight)
        self.num_blocks = max(1, int(num_blocks))
        self.steps_per_block = max(1, int(steps_per_block))
        self.pick_lowest_variance = bool(pick_lowest_variance)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.cfg: DictConfig = read_config(self.run_dir)
        fps_val = OmegaConf.select(self.cfg, "data.motion_loader.fps")
        if fps_val is None:
            # Dual-merged configs do not expose data.motion_loader.
            fps_val = OmegaConf.select(self.cfg, "data.motion_loader_t2m_latent_frame_text_aligned.fps")
        if fps_val is None:
            fps_val = OmegaConf.select(self.cfg, "data.motion_loader_t2m_latents.fps")
        self.fps: float = float(7.5 if fps_val is None else fps_val)
        self.featsname: str = str(getattr(self.cfg, "motion_features", "smplrifke"))
        self.abs_root: bool = abs_root

        self.ckpt_path = ckpt_path or self._find_latest_checkpoint()
        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        logger.info("Loading diffusion model from %s", self.ckpt_path)
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.diffusion = instantiate(self.cfg.diffusion)
        self.diffusion.load_state_dict(ckpt["state_dict"])
        self.diffusion.to(self.device)
        self.diffusion.eval()

        if not isinstance(self.diffusion, ActionPlanRectifiedFlow):
            raise ValueError(
                "ActionPlanSampler only supports ActionPlanRectifiedFlow; "
                f"got {type(self.diffusion).__name__}"
            )

        if sampling_timesteps is not None:
            self.diffusion.motion_steps = int(sampling_timesteps)
            self.diffusion.text_steps = int(sampling_timesteps)
            logger.info("Overriding diffusion steps: motion_steps=text_steps=%d", sampling_timesteps)

        self.stochastic_sampling = bool(
            getattr(self.diffusion, "stochastic_sampling", True)
        )
        self.variance_alpha = float(
            getattr(self.diffusion, "variance_alpha", 1.0)
        )
        self.sampling_temperature = float(
            getattr(self.diffusion, "sampling_temperature", 1.0)
        )

        modelpath = self.cfg.data.text_encoder.modelname
        mean_pooling = self.cfg.data.text_encoder.mean_pooling
        self.text_model = TextToEmb(
            modelpath=modelpath,
            mean_pooling=mean_pooling,
            device=str(self.device),
        )

    # --- Setup helpers ---

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Return path to most recently modified .ckpt in run_dir/logs/checkpoints."""
        ckpt_dir = os.path.join(self.run_dir, "logs", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return None
        candidates = []
        for name in os.listdir(ckpt_dir):
            if name.endswith(".ckpt"):
                path = os.path.join(ckpt_dir, name)
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    mtime = 0
                candidates.append((mtime, path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _build_text_embeddings(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode texts to embeddings; returns dict with 'x' and 'length' (or text_model keys)."""
        tx_emb = self.text_model(texts)
        if isinstance(tx_emb, torch.Tensor):
            return {
                "x": tx_emb[:, None],
                "length": torch.tensor(
                    [1 for _ in range(len(tx_emb))], device=self.device
                ),
            }
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in tx_emb.items()
        }

    def _resolve_duration(self, length: int) -> Tuple[int, int]:
        """Return (duration, effective_length). Pads to inference_length when set."""
        inference_length = getattr(self.diffusion, "inference_length", None)
        if inference_length is not None:
            duration = int(inference_length)
            effective_length = min(length, duration)
        else:
            duration = length
            effective_length = length
        return duration, effective_length

    def _get_sampling_params(self, y: Dict[str, Any]) -> Tuple[bool, float, float]:
        """Extract (stochastic, temperature, alpha) from y['infos']."""
        stochastic = bool(y["infos"].get("stochastic_sampling", self.stochastic_sampling))
        temperature = float(y["infos"].get("sampling_temperature", self.sampling_temperature))
        alpha = float(y["infos"].get("variance_alpha", self.variance_alpha))
        return stochastic, temperature, alpha

    # --- Phase 1: Text denoising ---

    @torch.no_grad()
    def _phase1_denoise_text(
        self,
        xt: torch.Tensor,
        y: Dict[str, Any],
        progress_bar=tqdm,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Denoise text dims (16:32) only; motion dims (0:16) stay noisy. Used for actionplan mode."""
        device = xt.device
        bs, duration, _ = xt.shape
        stochastic, temperature, alpha = self._get_sampling_params(y)

        tau_grid = torch.linspace(
            1.0, 0.0,
            self.diffusion.text_steps + 1,
            device=device,
            dtype=xt.dtype,
        )
        text_iter = list(zip(tau_grid[:-1], tau_grid[1:]))
        if progress_bar is not None:
            text_iter = progress_bar(text_iter, desc="ActionPlan Phase1 (text)")
        valid_mask = y.get("mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            zero_t = torch.zeros_like(valid_mask, dtype=xt.dtype)

        for tau, tau_next in text_iter:
            t_motion_val = torch.full(
                (bs, duration), 1.0, device=device, dtype=xt.dtype
            )
            t_text_val = torch.full(
                (bs, duration), float(tau), device=device, dtype=xt.dtype
            )
            t_text_next_val = torch.full(
                (bs, duration), float(tau_next), device=device, dtype=xt.dtype
            )
            if valid_mask is not None:
                # Keep padded frames clean (t=0) to mirror effective_length handling.
                t_motion_val = torch.where(valid_mask, t_motion_val, zero_t)
                t_text_val = torch.where(valid_mask, t_text_val, zero_t)
                t_text_next_val = torch.where(valid_mask, t_text_next_val, zero_t)
            t_ap = torch.stack([t_motion_val, t_text_val], dim=-1)
            v, logvar = self.diffusion._guided_model_output_actionplan(
                xt, y, t_ap, use_ema=use_ema
            )
            if stochastic:
                xt = self.diffusion._phase_step_stochastic(
                    xt, v, logvar, TEXT_SLICE,
                    t_text_val, t_text_next_val, alpha, temperature,
                )
            else:
                dt = float(tau) - float(tau_next)
                xt = self.diffusion._phase_step_ode(xt, v, TEXT_SLICE, dt)
        return xt

    def _phase1_pyramid_target_levels(
        self,
        effective_length: int,
        motion_steps: int,
        steps_per_block: Optional[int] = None,
    ) -> List[int]:
        """Initial level per frame for phase2_streaming. Frame 0=0; frame i>0 = min(motion_steps, i*steps_per_block)."""
        spb = max(1, int(self.steps_per_block if steps_per_block is None else steps_per_block))
        levels: List[int] = []
        for i in range(effective_length):
            if i == 0:
                levels.append(0)
            else:
                levels.append(min(motion_steps, spb * i))
        return levels

    def _build_block_to_frames(
        self,
        effective_length: int,
        num_blocks: Optional[int] = None,
        remaining_frames: Optional[List[int]] = None,
    ) -> Tuple[Dict[int, List[int]], int]:
        """Partition frames into contiguous coherent blocks."""
        remaining_set = (
            set(range(effective_length))
            if remaining_frames is None
            else set(remaining_frames)
        )
        num_blocks_for_sample = min(
            self.num_blocks if num_blocks is None else max(1, int(num_blocks)),
            max(1, effective_length),
        )
        block_size = max(
            1, (effective_length + num_blocks_for_sample - 1) // num_blocks_for_sample
        )
        num_blocks_out = (effective_length + block_size - 1) // block_size
        block_to_frames: Dict[int, List[int]] = {}
        for block_idx in range(num_blocks_out):
            start = block_idx * block_size
            end = min(start + block_size, effective_length)
            block_to_frames[block_idx] = [
                frame_idx for frame_idx in range(start, end) if frame_idx in remaining_set
            ]
        return block_to_frames, num_blocks_out

    def _phase1_pyramid_block_target_levels(
        self,
        effective_length: int,
        motion_steps: int,
        num_blocks: Optional[int] = None,
        steps_per_block: Optional[int] = None,
    ) -> List[int]:
        """Initial level per frame for block-wise streaming phase2."""
        spb = max(1, int(self.steps_per_block if steps_per_block is None else steps_per_block))
        block_to_frames, _ = self._build_block_to_frames(effective_length, num_blocks=num_blocks)
        levels = [motion_steps for _ in range(effective_length)]
        for block_idx, frames in block_to_frames.items():
            block_level = min(motion_steps, spb * block_idx)
            for frame_idx in frames:
                levels[frame_idx] = 0 if block_idx == 0 else block_level
        return levels

    # --- Phase 2: Motion denoising (pyramid / streaming) ---

    def _phase2_count_frame_pyramid_steps(
        self,
        initial_levels: Union[List[int], torch.Tensor],
        frame_order: Optional[List[int]] = None,
        frame_order_batches: Optional[List[List[int]]] = None,
    ) -> Optional[int]:
        """Count total micro-steps for frame-wise pyramid. Returns max(level + selection_delay) over frames."""
        if isinstance(initial_levels, torch.Tensor):
            levels = [int(v) for v in initial_levels.detach().cpu().tolist()]
        else:
            levels = [int(v) for v in initial_levels]

        positive = {idx for idx, level in enumerate(levels) if level > 0}
        if not positive:
            return 0

        spb = self.steps_per_block
        selection_delay: Dict[int, int] = {}
        selection_rank = 0

        if frame_order_batches is not None:
            for batch in frame_order_batches:
                added_any = False
                for frame_idx in batch:
                    if frame_idx in positive and frame_idx not in selection_delay:
                        selection_delay[frame_idx] = selection_rank * spb
                        added_any = True
                if added_any:
                    selection_rank += 1
        elif frame_order is not None:
            for frame_idx in frame_order:
                if frame_idx in positive and frame_idx not in selection_delay:
                    selection_delay[frame_idx] = selection_rank * spb
                    selection_rank += 1
        else:
            return None

        if len(selection_delay) != len(positive):
            return None

        return max(levels[frame_idx] + selection_delay[frame_idx] for frame_idx in positive)

    @torch.no_grad()
    def _phase1_denoise_text_and_first_motion(
        self,
        xt: torch.Tensor,
        y: Dict[str, Any],
        effective_length: int,
        progress_bar=tqdm,
        use_ema: bool = True,
        conditioning_frames: int = 0,
        conditioning_motion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise text and first motion frames together (streaming phase1). Adds frames 0,1,2,... every steps_per_block steps.
        conditioning_frames: if >0, clamp those frames to conditioning_motion after each step."""
        device = xt.device
        bs, duration, _ = xt.shape
        stochastic, temperature, alpha = self._get_sampling_params(y)

        motion_steps = self.diffusion.motion_steps
        steps = max(self.diffusion.text_steps, motion_steps)
        tau_grid = torch.linspace(
            1.0, 0.0, steps + 1, device=device, dtype=xt.dtype
        )
        steps_per_block = self.steps_per_block
        step_iter = list(zip(tau_grid[:-1], tau_grid[1:]))
        if progress_bar is not None:
            step_iter = progress_bar(
                step_iter, desc="ActionPlan Phase1 (streaming)"
            )
        valid_mask = y.get("mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            zero_t = torch.zeros_like(valid_mask, dtype=xt.dtype)

        for step_idx, (tau, tau_next) in enumerate(step_iter):
            max_active = step_idx // steps_per_block
            t_motion_val = torch.ones(bs, duration, device=device, dtype=xt.dtype)
            t_motion_next_val = torch.ones(
                bs, duration, device=device, dtype=xt.dtype
            )
            for idx in range(min(effective_length, duration)):
                if idx <= max_active:
                    t_motion_val[:, idx] = float(tau)
                    t_motion_next_val[:, idx] = float(tau_next)
                else:
                    t_motion_val[:, idx] = 1.0
                    t_motion_next_val[:, idx] = 1.0
            t_text_val = torch.full(
                (bs, duration), float(tau), device=device, dtype=xt.dtype
            )
            t_text_next_val = torch.full(
                (bs, duration), float(tau_next), device=device, dtype=xt.dtype
            )
            if valid_mask is not None:
                t_motion_val = torch.where(valid_mask, t_motion_val, zero_t)
                t_motion_next_val = torch.where(
                    valid_mask, t_motion_next_val, zero_t
                )
                t_text_val = torch.where(valid_mask, t_text_val, zero_t)
                t_text_next_val = torch.where(
                    valid_mask, t_text_next_val, zero_t
                )
            t_ap = torch.stack([t_motion_val, t_text_val], dim=-1)
            v, logvar = self.diffusion._guided_model_output_actionplan(
                xt, y, t_ap, use_ema=use_ema
            )
            if stochastic:
                xt = self.diffusion._phase_step_stochastic(
                    xt, v, logvar, TEXT_SLICE,
                    t_text_val, t_text_next_val, alpha, temperature,
                )
            else:
                dt = float(tau) - float(tau_next)
                xt = self.diffusion._phase_step_ode(xt, v, TEXT_SLICE, dt)
            if stochastic:
                x_next = self.diffusion._phase_step_stochastic(
                    xt, v, logvar, MOTION_SLICE,
                    t_motion_val, t_motion_next_val, alpha, temperature,
                )
            else:
                x_next = self._phase_step_ode_per_frame(
                    xt, v, MOTION_SLICE, t_motion_val, t_motion_next_val
                )
            xt = x_next
            if conditioning_frames > 0 and conditioning_motion is not None:
                n_cond = min(conditioning_frames, conditioning_motion.shape[1], xt.shape[1])
                xt[:, :n_cond, :MOTION_DIM] = conditioning_motion[..., :n_cond, :MOTION_DIM]
        return xt

    @torch.no_grad()
    def _phase1_denoise_text_and_first_block(
        self,
        xt: torch.Tensor,
        y: Dict[str, Any],
        effective_length: int,
        num_blocks: Optional[int] = None,
        progress_bar=tqdm,
        use_ema: bool = True,
        conditioning_frames: int = 0,
        conditioning_motion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise text and the first coherent block together, then activate later blocks."""
        device = xt.device
        bs, duration, _ = xt.shape
        stochastic, temperature, alpha = self._get_sampling_params(y)

        motion_steps = self.diffusion.motion_steps
        steps = max(self.diffusion.text_steps, motion_steps)
        tau_grid = torch.linspace(
            1.0, 0.0, steps + 1, device=device, dtype=xt.dtype
        )
        steps_per_block = self.steps_per_block
        step_iter = list(zip(tau_grid[:-1], tau_grid[1:]))
        if progress_bar is not None:
            step_iter = progress_bar(
                step_iter, desc="ActionPlan Phase1 (streaming_block)"
            )
        valid_mask = y.get("mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            zero_t = torch.zeros_like(valid_mask, dtype=xt.dtype)

        block_to_frames, _ = self._build_block_to_frames(effective_length, num_blocks=num_blocks)

        for step_idx, (tau, tau_next) in enumerate(step_iter):
            max_active_block = step_idx // steps_per_block
            active_frames: Set[int] = set()
            for block_idx, frames in block_to_frames.items():
                if block_idx <= max_active_block:
                    active_frames.update(frames)

            t_motion_val = torch.ones(bs, duration, device=device, dtype=xt.dtype)
            t_motion_next_val = torch.ones(
                bs, duration, device=device, dtype=xt.dtype
            )
            for idx in range(min(effective_length, duration)):
                if idx in active_frames:
                    t_motion_val[:, idx] = float(tau)
                    t_motion_next_val[:, idx] = float(tau_next)
                else:
                    t_motion_val[:, idx] = 1.0
                    t_motion_next_val[:, idx] = 1.0
            t_text_val = torch.full(
                (bs, duration), float(tau), device=device, dtype=xt.dtype
            )
            t_text_next_val = torch.full(
                (bs, duration), float(tau_next), device=device, dtype=xt.dtype
            )
            if valid_mask is not None:
                t_motion_val = torch.where(valid_mask, t_motion_val, zero_t)
                t_motion_next_val = torch.where(
                    valid_mask, t_motion_next_val, zero_t
                )
                t_text_val = torch.where(valid_mask, t_text_val, zero_t)
                t_text_next_val = torch.where(
                    valid_mask, t_text_next_val, zero_t
                )

            t_ap = torch.stack([t_motion_val, t_text_val], dim=-1)
            v, logvar = self.diffusion._guided_model_output_actionplan(
                xt, y, t_ap, use_ema=use_ema
            )
            if stochastic:
                xt = self.diffusion._phase_step_stochastic(
                    xt, v, logvar, TEXT_SLICE,
                    t_text_val, t_text_next_val, alpha, temperature,
                )
            else:
                dt = float(tau) - float(tau_next)
                xt = self.diffusion._phase_step_ode(xt, v, TEXT_SLICE, dt)
            if stochastic:
                x_next = self.diffusion._phase_step_stochastic(
                    xt, v, logvar, MOTION_SLICE,
                    t_motion_val, t_motion_next_val, alpha, temperature,
                )
            else:
                x_next = self._phase_step_ode_per_frame(
                    xt, v, MOTION_SLICE, t_motion_val, t_motion_next_val
                )
            xt = x_next
            if conditioning_frames > 0 and conditioning_motion is not None:
                n_cond = min(conditioning_frames, conditioning_motion.shape[1], xt.shape[1])
                xt[:, :n_cond, :MOTION_DIM] = conditioning_motion[..., :n_cond, :MOTION_DIM]
        return xt

    def _frame_variances(
        self,
        xt: torch.Tensor,
        y: Dict[str, Any],
        tau_motion: torch.Tensor,
        effective_length: int,
    ) -> torch.Tensor:
        """Per-frame variance over motion dims (0:16). Used by _phase2_pyramid for variance-based frame picking."""
        device = xt.device
        bs, duration, _ = xt.shape
        t_text_val = torch.zeros(bs, duration, device=device, dtype=xt.dtype)
        t_ap = torch.stack([tau_motion, t_text_val], dim=-1)
        v, logvar = self.diffusion._guided_model_output_actionplan(
            xt, y, t_ap, use_ema=True
        )
        if logvar is not None:
            var = torch.exp(
                torch.clamp(logvar[..., MOTION_SLICE], min=-30.0, max=20.0)
            )
            out = var.mean(dim=-1)[0]
        else:
            out = xt[0, :, MOTION_SLICE].var(dim=-1, unbiased=False)
        return out[:effective_length]

    def _build_motion_tau_from_levels(
        self,
        bs: int,
        duration: int,
        effective_length: int,
        level_state: torch.Tensor,
        tau_grid: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build per-frame t_motion from level_state. High level -> tau=1 (noisy), level=0 -> tau=0 (clean)."""
        motion_steps = tau_grid.shape[0] - 1
        t = torch.ones(bs, duration, device=device, dtype=dtype)
        if effective_length < duration:
            t[:, effective_length:] = 0.0
        for idx in range(effective_length):
            lv = int(level_state[idx].item())
            if lv <= 0:
                t[:, idx] = 0.0
            else:
                # high level (motion_steps) -> tau 1.0 (noisy), level 1 -> tau_grid[motion_steps-1] (~0)
                tau_idx = min(motion_steps, motion_steps - lv)
                t[:, idx] = tau_grid[tau_idx].item()
        return t

    @staticmethod
    def _phase_step_ode_per_frame(
        xt: torch.Tensor,
        v: torch.Tensor,
        dim_slice: slice,
        t_motion_val: torch.Tensor,
        t_motion_next_val: torch.Tensor,
    ) -> torch.Tensor:
        """ODE step with per-frame dt. Used when motion frames have different tau values."""
        v_stream = v[..., dim_slice]
        dt = (t_motion_val - t_motion_next_val).unsqueeze(-1)
        stream_next = xt[..., dim_slice] - v_stream * dt
        parts = [xt[..., : dim_slice.start], stream_next, xt[..., dim_slice.stop :]]
        return torch.cat([p for p in parts if p.numel() > 0], dim=-1)

    @torch.no_grad()
    def _phase2_step_frames(
        self,
        xt: torch.Tensor,
        y: Dict[str, Any],
        level_state: torch.Tensor,
        frame_indices: List[int],
        duration: int,
        effective_length: int,
        tau_grid: torch.Tensor,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """One pyramid micro-step: denoise frames in frame_indices, decrement their level_state."""
        device = xt.device
        bs = xt.shape[0]
        stochastic, temperature, alpha = self._get_sampling_params(y)

        next_level = level_state.clone()
        for idx in frame_indices:
            if idx < effective_length and level_state[idx] > 0:
                next_level[idx] = level_state[idx] - 1

        t_motion_val = self._build_motion_tau_from_levels(
            bs, duration, effective_length, level_state, tau_grid, device, xt.dtype
        )
        t_motion_next_val = self._build_motion_tau_from_levels(
            bs, duration, effective_length, next_level, tau_grid, device, xt.dtype
        )
        t_text_val = torch.zeros(bs, duration, device=device, dtype=xt.dtype)
        t_ap = torch.stack([t_motion_val, t_text_val], dim=-1)
        v, logvar = self.diffusion._guided_model_output_actionplan(
            xt, y, t_ap, use_ema=use_ema
        )

        if stochastic:
            x_next = self.diffusion._phase_step_stochastic(
                xt, v, logvar, MOTION_SLICE,
                t_motion_val, t_motion_next_val, alpha, temperature,
            )
        else:
            x_next = self._phase_step_ode_per_frame(
                xt, v, MOTION_SLICE, t_motion_val, t_motion_next_val
            )

        # Same update pattern as KeyframeSampler:
        # only frames whose level changes are updated.
        keep_mask = (level_state.unsqueeze(0) == next_level.unsqueeze(0)).unsqueeze(-1)
        xt = torch.where(keep_mask, xt, x_next)
        level_state.copy_(next_level)
        return xt

    @torch.no_grad()
    def _phase2_streaming(
        self,
        xt: torch.Tensor,
        y: Dict[str, Any],
        duration: int,
        effective_length: int,
        conditioning_frames: int = 0,
        on_frame_ready: Optional[Callable[[int, np.ndarray], None]] = None,
        progress_bar=tqdm,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Pyramid denoise remaining frames (streaming phase2). Frame 0 done from phase1. Calls on_frame_ready as each frame completes."""
        if effective_length <= 1:
            if on_frame_ready is not None and effective_length == 1:
                x_out = self.diffusion.motion_normalizer.inverse(xt)
                latents_16 = x_out[0, :1, :MOTION_DIM].detach().cpu().numpy()
                on_frame_ready(0, latents_16)
            return xt
        device = xt.device
        motion_steps = self.diffusion.motion_steps
        target_levels = self._phase1_pyramid_target_levels(effective_length, motion_steps)
        level_state = torch.tensor(
            target_levels + [0] * (duration - effective_length),
            dtype=torch.long,
            device=device,
        )
        if conditioning_frames > 0:
            n_cond = min(conditioning_frames, effective_length)
            level_state[:n_cond] = 0
        if effective_length < duration:
            level_state[effective_length:] = 0
        remaining_frames = [i for i in range(effective_length) if level_state[i] > 0]
        remaining_set = set(remaining_frames)
        tau_grid = torch.linspace(1.0, 0.0, motion_steps + 1, device=device, dtype=xt.dtype)

        picked: List[int] = []
        frame_order = list(range(1, effective_length))
        frame_order_idx = 0
        total_steps = self._phase2_count_frame_pyramid_steps(level_state, frame_order=frame_order)
        pbar = progress_bar(total=total_steps, desc="ActionPlan Phase2 (streaming)") if progress_bar else None

        frames_ready: Set[int] = set()
        if on_frame_ready is not None:
            for i in range(effective_length):
                if level_state[i].item() == 0:
                    frames_ready.add(i)
                    x_out = self.diffusion.motion_normalizer.inverse(xt)
                    latents_16 = x_out[0, : i + 1, :MOTION_DIM].detach().cpu().numpy()
                    on_frame_ready(i, latents_16)

        while level_state[remaining_frames].max().item() > 0:
            if len(picked) < len(remaining_frames):
                while frame_order_idx < len(frame_order):
                    new_idx = frame_order[frame_order_idx]
                    frame_order_idx += 1
                    if new_idx in remaining_set and new_idx not in picked and level_state[new_idx] > 0:
                        picked.append(new_idx)
                        break

            for _ in range(self.steps_per_block):
                if level_state[remaining_frames].max().item() <= 0:
                    break
                xt = self._phase2_step_frames(
                    xt, y, level_state, picked, duration, effective_length,
                    tau_grid, use_ema=use_ema,
                )
                if pbar is not None:
                    pbar.update(1)
                if on_frame_ready is not None:
                    for i in range(effective_length):
                        if level_state[i].item() == 0 and i not in frames_ready:
                            frames_ready.add(i)
                            x_out = self.diffusion.motion_normalizer.inverse(xt)
                            latents_16 = x_out[0, : i + 1, :MOTION_DIM].detach().cpu().numpy()
                            on_frame_ready(i, latents_16)
        if pbar is not None:
            pbar.close()
        return xt

    @torch.no_grad()
    def _phase2_streaming_block(
        self,
        xt: torch.Tensor,
        y: Dict[str, Any],
        duration: int,
        effective_length: int,
        num_blocks: Optional[int] = None,
        conditioning_frames: int = 0,
        on_block_ready: Optional[Callable[[int, np.ndarray], None]] = None,
        progress_bar=tqdm,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Block-wise streaming phase2: selected blocks are denoised coherently."""
        if effective_length <= 0:
            return xt
        device = xt.device
        motion_steps = self.diffusion.motion_steps
        target_levels = self._phase1_pyramid_block_target_levels(
            effective_length, motion_steps, num_blocks=num_blocks
        )
        level_state = torch.tensor(
            target_levels + [0] * (duration - effective_length),
            dtype=torch.long,
            device=device,
        )
        if conditioning_frames > 0:
            n_cond = min(conditioning_frames, effective_length)
            level_state[:n_cond] = 0
        if effective_length < duration:
            level_state[effective_length:] = 0

        block_to_frames, num_blocks_out = self._build_block_to_frames(
            effective_length, num_blocks=num_blocks
        )
        remaining_frames = [i for i in range(effective_length) if level_state[i] > 0]
        if not remaining_frames:
            if on_block_ready is not None:
                x_out = self.diffusion.motion_normalizer.inverse(xt)
                latents_16 = x_out[0, :effective_length, :MOTION_DIM].detach().cpu().numpy()
                for block_idx, frames in block_to_frames.items():
                    if frames:
                        on_block_ready(block_idx, latents_16[: frames[-1] + 1])
            return xt

        tau_grid = torch.linspace(
            1.0, 0.0, motion_steps + 1, device=device, dtype=xt.dtype
        )
        selected_blocks: List[int] = []
        block_order = list(range(1, num_blocks_out))
        block_order_idx = 0
        total_steps = motion_steps * max(
            1, len([frames for frames in block_to_frames.values() if frames])
        )
        pbar = progress_bar(
            total=total_steps, desc="ActionPlan Phase2 (streaming_block)"
        ) if progress_bar else None

        ready_blocks: Set[int] = set()

        def emit_ready_blocks() -> None:
            if on_block_ready is None:
                return
            x_out = self.diffusion.motion_normalizer.inverse(xt)
            latents_16 = x_out[0, :effective_length, :MOTION_DIM].detach().cpu().numpy()
            for block_idx, frames in block_to_frames.items():
                if not frames or block_idx in ready_blocks:
                    continue
                if all(level_state[frame_idx].item() == 0 for frame_idx in frames):
                    ready_blocks.add(block_idx)
                    on_block_ready(block_idx, latents_16[: frames[-1] + 1])

        emit_ready_blocks()

        while level_state[remaining_frames].max().item() > 0:
            if len(selected_blocks) < len(block_order):
                while block_order_idx < len(block_order):
                    new_block = block_order[block_order_idx]
                    block_order_idx += 1
                    frames = block_to_frames.get(new_block, [])
                    if not frames:
                        continue
                    if all(level_state[idx] <= 0 for idx in frames):
                        continue
                    selected_blocks.append(new_block)
                    break

            active_frames: List[int] = []
            for block_idx in selected_blocks:
                for frame_idx in block_to_frames[block_idx]:
                    if level_state[frame_idx] > 0:
                        active_frames.append(frame_idx)

            for _ in range(self.steps_per_block):
                if level_state[remaining_frames].max().item() <= 0:
                    break
                if active_frames:
                    xt = self._phase2_step_frames(
                        xt, y, level_state, active_frames, duration, effective_length,
                        tau_grid, use_ema=use_ema,
                    )
                if pbar is not None:
                    pbar.update(1)
                emit_ready_blocks()

        if pbar is not None:
            pbar.close()
        return xt

    @torch.no_grad()
    def _phase2_pyramid(
        self,
        xt: torch.Tensor,
        y: Dict[str, Any],
        duration: int,
        effective_length: int,
        remaining_frames: List[int],
        frame_order: Optional[List[int]] = None,
        frame_order_batches: Optional[List[List[int]]] = None,
        pick_lowest: Optional[bool] = None,
        keyframe_set: Optional[Set[int]] = None,
        progress_bar=tqdm,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Pyramid denoise (actionplan phase2). Add frames by frame_order or variance; step picked set for steps_per_block.
        frame_order: add one frame at a time in this order. If None, pick by variance (pick_lowest).
        keyframe_set: frames already at level 0 (denoised).
        """
        device = xt.device
        bs = xt.shape[0]
        pick_low = self.pick_lowest_variance if pick_lowest is None else pick_lowest
        remaining_set = set(remaining_frames)
        motion_steps = self.diffusion.motion_steps
        tau_grid = torch.linspace(
            1.0, 0.0, motion_steps + 1, device=device, dtype=xt.dtype
        )
        level_state = torch.full(
            (duration,), motion_steps, dtype=torch.long, device=device
        )
        if effective_length < duration:
            level_state[effective_length:] = 0
        if keyframe_set:
            for idx in keyframe_set:
                if idx < duration:
                    level_state[idx] = 0

        picked: List[int] = []
        frame_order_idx = 0
        frame_order_batches_idx = 0
        total_steps = self._phase2_count_frame_pyramid_steps(
            level_state,
            frame_order=frame_order,
            frame_order_batches=frame_order_batches,
        )
        pbar = progress_bar(total=total_steps, desc="ActionPlan Phase2 (pyramid)") if progress_bar else None

        while level_state[remaining_frames].max().item() > 0:
            if len(picked) < len(remaining_frames):
                if frame_order_batches is not None:
                    # Add entire batch (e.g. [i-1, i+1]) at once
                    while frame_order_batches_idx < len(frame_order_batches):
                        batch = frame_order_batches[frame_order_batches_idx]
                        frame_order_batches_idx += 1
                        added_any = False
                        for new_idx in batch:
                            if new_idx in remaining_set and new_idx not in picked and level_state[new_idx] > 0:
                                picked.append(new_idx)
                                added_any = True
                        if added_any:
                            break
                    else:
                        frame_order_batches_idx = len(frame_order_batches)  # exhausted
                elif frame_order is not None:
                    while frame_order_idx < len(frame_order):
                        new_idx = frame_order[frame_order_idx]
                        frame_order_idx += 1
                        if new_idx in remaining_set and new_idx not in picked and level_state[new_idx] > 0:
                            picked.append(new_idx)
                            break
                else:
                    tau_motion = self._build_motion_tau_from_levels(
                        bs, duration, effective_length, level_state, tau_grid, device, xt.dtype
                    )
                    variances = self._frame_variances(xt, y, tau_motion, effective_length)
                    fill = float("inf") if pick_low else float("-inf")
                    masked = variances.clone()
                    masked[effective_length:] = fill
                    for idx in range(effective_length):
                        if idx not in remaining_set:
                            masked[idx] = fill
                    for idx in picked:
                        masked[idx] = fill
                    if pick_low:
                        new_idx = int(torch.argmin(masked[:effective_length]).item())
                    else:
                        new_idx = int(torch.argmax(masked[:effective_length]).item())
                    if new_idx in remaining_set and new_idx not in picked and level_state[new_idx] > 0:
                        picked.append(new_idx)

            for _ in range(self.steps_per_block):
                if level_state[remaining_frames].max().item() <= 0:
                    break
                xt = self._phase2_step_frames(
                    xt, y, level_state, picked, duration, effective_length,
                    tau_grid, use_ema=use_ema,
                )
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()
        return xt

    # --- Public API ---

    @torch.no_grad()
    def sample(
        self,
        text: str,
        seconds: Optional[float] = None,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
        fps: Optional[float] = None,
        conditioning_latents: Optional[Union[np.ndarray, torch.Tensor]] = None,
        conditioning_frames: int = 0,
        on_frame_ready: Optional[Callable[[int, np.ndarray], None]] = None,
        on_block_ready: Optional[Callable[[int, np.ndarray], None]] = None,
        num_blocks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate motion from text. Returns dict with 'features' (16-dim latents), 'length', 'fps'."""
        fps_val = float(fps) if fps is not None else self.fps
        if seconds is None or seconds <= 0:
            seconds = 4.0
        req_length = int(round(fps_val * float(seconds)))
        if conditioning_latents is not None and conditioning_frames > 0:
            req_length = conditioning_frames + max(1, req_length)

        duration, effective_length = self._resolve_duration(req_length)
        if effective_length < req_length:
            logger.warning(
                "Requested length %s exceeds inference_length %s; truncating to %s.",
                req_length, duration, effective_length,
            )

        infos: Dict[str, Any] = {
            "all_lengths": [effective_length],
            "all_texts": [text],
            "featsname": self.featsname,
            "guidance_weight": float(self.guidance_weight),
            "stochastic_sampling": self.stochastic_sampling,
            "variance_alpha": self.variance_alpha,
            "sampling_temperature": self.sampling_temperature,
        }
        tx = self._build_text_embeddings([text])
        tx_uncond = self._build_text_embeddings([""])

        mask = torch.zeros((1, duration), device=self.device, dtype=torch.bool)
        if effective_length > 0:
            mask[:, :effective_length] = True
        y = {
            "length": [effective_length],
            "mask": mask,
            "tx": self.diffusion.prepare_tx_emb(tx),
            "tx_uncond": self.diffusion.prepare_tx_emb(tx_uncond),
            "infos": infos,
        }

        nfeats = self.diffusion.denoiser.nfeats
        xt = torch.randn((1, duration, nfeats), device=self.device)

        cond_normalized = None
        n_cond = 0
        if conditioning_latents is not None and conditioning_frames > 0:
            n_cond = min(conditioning_frames, effective_length)
            cond = conditioning_latents
            if isinstance(cond, np.ndarray):
                cond = torch.from_numpy(cond).float().to(self.device)
            else:
                cond = cond.to(self.device).float()
            if cond.dim() == 2:
                cond = cond.unsqueeze(0)
            if cond.shape[-1] == MOTION_DIM:
                cond_full = torch.zeros(
                    1, n_cond, nfeats, device=self.device, dtype=cond.dtype
                )
                cond_full[..., :MOTION_DIM] = cond[:, :n_cond, :MOTION_DIM]
                cond = cond_full
            else:
                cond = cond[:, :n_cond]
            cond_normalized = self.diffusion.motion_normalizer(cond)
            xt[:, :n_cond, :MOTION_DIM] = cond_normalized[:, :, :MOTION_DIM]

        num_blocks_to_use = self.num_blocks if num_blocks is None else max(1, int(num_blocks))

        if self.mode in ("actionplan", "joint"):
            xt = self._phase1_denoise_text(xt, y, progress_bar=tqdm, use_ema=True)
        elif self.mode == "streaming_block":
            xt = self._phase1_denoise_text_and_first_block(
                xt, y, effective_length, num_blocks=num_blocks_to_use,
                progress_bar=tqdm, use_ema=True,
                conditioning_frames=n_cond,
                conditioning_motion=cond_normalized,
            )
        else:
            xt = self._phase1_denoise_text_and_first_motion(
                xt, y, effective_length, progress_bar=tqdm, use_ema=True,
                conditioning_frames=n_cond,
                conditioning_motion=cond_normalized,
            )

        remaining_frames = list(range(effective_length))

        if self.mode == "streaming":
            xt = self._phase2_streaming(
                xt, y, duration, effective_length,
                conditioning_frames=n_cond,
                on_frame_ready=on_frame_ready,
                progress_bar=tqdm, use_ema=True,
            )
        elif self.mode == "streaming_block":
            xt = self._phase2_streaming_block(
                xt, y, duration, effective_length,
                num_blocks=num_blocks_to_use,
                conditioning_frames=n_cond,
                on_block_ready=on_block_ready,
                progress_bar=tqdm, use_ema=True,
            )
        elif self.mode == "joint":
            # All motion latents active from the start -> fully parallel denoising.
            xt = self._phase2_pyramid(
                xt, y, duration, effective_length, remaining_frames,
                frame_order_batches=[list(remaining_frames)],
                progress_bar=tqdm,
                use_ema=True,
            )
        else:
            order = list(remaining_frames)
            np.random.shuffle(order)
            xt = self._phase2_pyramid(
                xt, y, duration, effective_length, remaining_frames,
                frame_order=order,
                pick_lowest=self.pick_lowest_variance,
                progress_bar=tqdm,
                use_ema=True,
            )

        x_out = self.diffusion.motion_normalizer.inverse(masked(xt, mask))
        xstart_full = x_out[0, :effective_length].detach().cpu().numpy()
        # Eval_272 and TAE expect 16-dim motion latents only (dims 0:16); text dims 16:32 are not used for 272 metrics.
        xstart_np = xstart_full[..., :MOTION_DIM].copy()

        return {
            "features": xstart_np,
            "length": effective_length,
            "fps": fps_val,
        }

    def sample_streaming(
        self,
        text: str,
        seconds: Optional[float] = None,
        fps: Optional[float] = None,
        conditioning_latents: Optional[Union[np.ndarray, torch.Tensor]] = None,
        conditioning_frames: int = 0,
        num_blocks: Optional[int] = None,
        progress_bar=tqdm,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate motion with real-time streaming.

        Yields:
          frame_ready: (frame_idx, latents) as each frame completes; latents shape (frame_idx+1, 16)
          block_ready: (block_idx, latents) as each block completes; latents shape (block_end+1, 16)
          complete: final latents and length
        """
        if self.mode not in {"streaming", "streaming_block"}:
            raise ValueError("sample_streaming requires mode in {'streaming', 'streaming_block'}")
        q: queue.Queue = queue.Queue()

        def on_frame_ready_cb(frame_idx: int, latents_16: np.ndarray) -> None:
            q.put({"type": "frame_ready", "frame_idx": frame_idx, "latents": latents_16.copy()})

        def on_block_ready_cb(block_idx: int, latents_16: np.ndarray) -> None:
            q.put({"type": "block_ready", "block_idx": block_idx, "latents": latents_16.copy()})

        def run_sample() -> None:
            try:
                result = self.sample(
                    text=text,
                    seconds=seconds,
                    fps=fps,
                    conditioning_latents=conditioning_latents,
                    conditioning_frames=conditioning_frames,
                    on_frame_ready=on_frame_ready_cb if self.mode == "streaming" else None,
                    on_block_ready=on_block_ready_cb if self.mode == "streaming_block" else None,
                    num_blocks=num_blocks,
                )
                q.put({"type": "complete", "latents": result["features"], "length": result["length"]})
            except Exception as e:
                q.put({"type": "error", "error": e})
            finally:
                q.put(None)

        t = threading.Thread(target=run_sample)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            if item.get("type") == "error":
                raise item["error"]
            yield item
        t.join()
