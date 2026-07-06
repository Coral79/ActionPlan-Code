"""Evaluate ActionPlan on the HumanML3D-272 test set (paper Tables 1 and 4).

Uses MotionStreamer's evaluation protocol with their pretrained TMR-based
evaluator (models/Evaluator_272). Requires the 272-dim dataset; download it
with: python prepare/download_streamer272_data.py

Samplers (all use the same checkpoint, only the sampling schedule differs):
    offline     ActionPlan-Offline  (Table 1): random pyramid, K=2 overlap
    streaming   ActionPlan-Streaming (Table 1): text+first latent together,
                then raster-order pyramid, K=2
    parallel    Table 4 "Fully overlap (Parallel)": all latents denoised jointly
    offline_s5 / offline_s10 / offline_s15
                Table 4 "K-step non-overlap" rows
    offline_s25 Table 4 "Fully non-overlap (Random)": each latent fully
                denoised before the next is activated (25 = all flow steps)

Paper settings: guidance 5.5, split test.txt, seeds 123..(123+reps-1).
The paper reports the mean over ~20 replications; pass --replication_times 20
to reproduce (or run multiple times with different --seed and average).

Example (Table 1 offline):
    python eval.py --sampler offline --replication_times 20
"""

import argparse
import os
import sys
from types import SimpleNamespace

from eval.eval import print_metrics, run_eval

HUMANML3D_272_DIR = "datasets/motions/humanml3d_272"
EVALUATOR_CKPT = "models/Evaluator_272/epoch=99.ckpt"

DEFAULT_RUN_DIR = "outputs/actionplan"
DEFAULT_CKPT = os.path.join(DEFAULT_RUN_DIR, "logs", "checkpoints", "latest-epoch=9999.ckpt")

# Paper sampler configurations. "mode" and "steps_per_block" are passed to
# ActionPlanSampler; steps_per_block is the overlap window K from Table 4.
SAMPLER_CONFIGS = {
    "offline": {"name": "offline", "mode": "actionplan", "steps_per_block": 2},
    "streaming": {"name": "streaming", "mode": "streaming", "steps_per_block": 2},
    "parallel": {"name": "parallel", "mode": "joint"},
    "offline_s5": {"name": "offline_s5", "mode": "actionplan", "steps_per_block": 5},
    "offline_s10": {"name": "offline_s10", "mode": "actionplan", "steps_per_block": 10},
    "offline_s15": {"name": "offline_s15", "mode": "actionplan", "steps_per_block": 15},
    "offline_s25": {"name": "offline_s25", "mode": "actionplan", "steps_per_block": 25},
}


def main():
    parser = argparse.ArgumentParser(
        description="272-dim evaluation (MotionStreamer protocol).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sampler", type=str, default="offline",
        choices=list(SAMPLER_CONFIGS.keys()) + ["all"],
        help="Sampling schedule to evaluate ('all' runs every paper schedule).",
    )
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR, help="Model run directory.")
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT, help="Model checkpoint.")
    parser.add_argument("--guidance_weight", type=float, default=5.5, help="Classifier-free guidance weight (paper: 5.5).")
    parser.add_argument("--split_file", type=str, default="test.txt", help="Split file under datasets/motions/humanml3d_272/split/.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed (paper: 123).")
    parser.add_argument("--replication_times", type=int, default=1, help="Replications for mean ± 95%% CI (paper: ~20).")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of test samples (default: full test set).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the evaluator.")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPUs for parallel generation (cuda:0 is reserved for the evaluator).")
    parser.add_argument("--sampling_timesteps", type=int, default=None, help="Override rectified-flow steps (default: from config, 25).")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save metrics JSONs (default: <run_dir>/eval_results).")
    cli = parser.parse_args()

    if cli.sampler == "all":
        samplers = list(SAMPLER_CONFIGS.values())
    else:
        samplers = [SAMPLER_CONFIGS[cli.sampler]]

    for path, hint in [
        (HUMANML3D_272_DIR + "/motion_data", "python prepare/download_streamer272_data.py"),
        (EVALUATOR_CKPT, "python prepare/download_dependencies.py"),
        (cli.ckpt_path, "python prepare/download_dependencies.py"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run: {hint}")
            sys.exit(1)

    args = SimpleNamespace(
        humanml3d_272_dir=HUMANML3D_272_DIR,
        evaluator_ckpt=EVALUATOR_CKPT,
        run_dir=cli.run_dir,
        ckpt_path=cli.ckpt_path,
        guidance_weight=cli.guidance_weight,
        sampling_timesteps=cli.sampling_timesteps,
        split_file=cli.split_file,
        seed=cli.seed,
        replication_times=cli.replication_times,
        num_samples=cli.num_samples,
        batch_size=cli.batch_size,
        num_gpus=cli.num_gpus,
        output_dir=cli.output_dir,
        unit_length=4,
        diversity_times=300,
        device=None,
        samplers=samplers,
    )

    results = run_eval(args)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print_metrics("GT (Real)", results["gt_metrics"])
    for sampler_name, sampler_result in results["sampler_metrics"].items():
        print_metrics(sampler_name, sampler_result["metrics"])


if __name__ == "__main__":
    main()
