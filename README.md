# ActionPlan: Future-Aware Streaming Motion Synthesis via Frame-Level Action Planning

<p align="center">
  <a href="https://coral79.github.io/ActionPlan/"><b>[🌐 Project Page]</b></a>
  <a href="#"><b>[📄 arXiv]</b></a>
</p>

This is the official repository for **ActionPlan**.

---

## 🚀 News
- **[18/06/2026]** ActionPlan got accepted to ECCV 2026!

- **[25/03/2026]** Inference code for the online demo released.

- **[March 2026]** Our paper has been submitted to arXiv!

## 📦 Release Plan
- ✅ **Inference Code**: Run the real-time streaming demo yourself.
- ✅ **Model Weights**: Pre-trained checkpoints for ActionPlan.
- ✅ **Training Code**: Full training pipeline.
- ✅ **Evaluation Code**: Evaluation pipeline.

## 📄 Abstract
We present **ActionPlan**, a unified motion diffusion framework that bridges real-time streaming with high-quality offline generation within a single model. The core idea is to introduce a *per-frame action plan*: the model predicts frame-level text latents that act as dense semantic anchors throughout denoising, and uses them to denoise the full motion sequence with combined semantic and motion cues.

To support this structured workflow, we design latent-specific diffusion steps, allowing each motion latent to be denoised independently and sampled in flexible orders at inference. As a result, ActionPlan can run in a history-conditioned, future-aware mode for real-time streaming, while also supporting high-quality offline generation.

The same mechanism further enables zero-shot motion editing and in-betweening without additional models. Experiments demonstrate that our real-time streaming is **5.25× faster** while achieving **18% motion quality improvement** over the best previous method in terms of FID.

---


## Install Environment

```bash
# Create venv with Python 3.10 (use pyenv, conda, or system python)
python3.10 -m venv venv
source venv/bin/activate   # Linux/macOS
# or: ActionPlan-Code\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git@main --prefer-binary
pip install --no-build-isolation chumpy
pip install -r requirements.txt
```

---

## Data Preparation

Download and extract actionplan dependencies (annotations, embeddings, stats, etc.):

```bash
python prepare/download_dependencies.py
```

**Directory Structure**

```
ActionPlan-Code/
├── generate.py       # Motion generation
├── render.py         # Render motion to video
├── requirements.txt  # Python dependencies
├── demo/             # Interactive streaming demo (server, frontend, sampler)
├── deps/             # SMPL-H and joint assets (used by rendering / data prep)
├── outputs/          # Checkpoints and generation outputs
├── prepare/          # Data download scripts
├── src/              # Model, data, samplers, renderer, tools
├── models/           # Bundled weights & eval code (Evaluator_272, Causal_TAE, clip_autoencoder)
└── datasets/         # Annotations, splits, frame-motion latents
    ├── embeddings/
    ├── motions/
    └── stats/
```

---


## Interactive Demo Interface

### Prerequisites

Make sure you have **Node.js** installed.

You can download it from: [https://nodejs.org](https://nodejs.org)

Verify the installation:

```bash
node -v
npm -v
```

### Run the Demo

To start the interactive demo interface, run:

```bash
bash demo/run.sh
```

---

## Offline Sampling

**Text-to-Motion Generation**

To sample a random prompt from the humanml3d test set (make sure to go through data preparation first):

```shell
python generate.py -n 1
```

To sample a custom prompt:

```shell
python generate.py -t "a person walks forward"
```

---

## Streaming Sampling

**Streaming Text-to-Motion Generation**

Interactive mode: enter prompts one by one; each new sequence is conditioned on the previous. SMPL params are written to disk as each frame is denoised for real-time access by other processes.

```shell
python generate.py --streaming
```

**Output layout (per sequence):**

- `streaming_smpl_current.npz` — live SMPL (poses, trans, joints) during denoise
- `{prompt}_latents.npy`, `{prompt}_decoded272.npy`, `{prompt}.npz` — final latent, decoded, and SMPL params

**Commands:** `/q` quit, `/rq` render full sequence and quit, `/reset` start new session.

Outputs go to `outputs/actionplan/generations/generate_streaming_latent/session_YYYYMMDD_HHMMSS/`

### SONIC + Unitree G1 (ZMQ pose stream)

Publishes a **ZMQ protocol v3** “pose” stream intended for **[SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://nvlabs.github.io/GEAR-SONIC/)**  when you run their stack with a **ZMQ motion input** (e.g. reference tracking on a **Unitree G1**). 

```shell
python generate.py --streaming --g1 --g1-host '*' --g1-port 5556 --g1-topic pose --g1-hz 50
```

`/reset` clears streaming session state and G1 continuity; exiting the script stops the ZMQ publisher.

---

## Rendering

Render SMPL mesh videos from `.npy` files. Supports three input formats (auto-detected by feature dimension):


| Dim | Format                | Description                                |
| --- | --------------------- | ------------------------------------------ |
| 16  | Motion latents only   | TAE-decoded to 272-dim, then SMPL          |
| 32  | Motion + text latents | First 16 dims (motion) used; same as above |
| 272 | Decoded motion        | Directly converted to SMPL vertices        |


```shell
# Basic usage (output: input_path.mp4)
python render.py path/to/motion.npy

# With options
python render.py path/to/motion.npy --out_path output.mp4 --fps 30
```

**Options:** `--out_path`, `--fps` (default: 30), `--ext` (default: mp4), `--tae_checkpoint`, `--device`, `--y_is_z_axis`

---

## Training

Train the final ActionPlan model reported in the paper. Besides the standard dependencies (`prepare/download_dependencies.py`), training needs the 16-dim TAE latents of the HumanML3D-272 dataset:

```bash
# Download the 272-dim HumanML3D data and encode it to 16-dim TAE latents
python prepare/download_streamer272_data.py
```

Then start training (config: `configs/train_actionplan.yaml`, defaults to 4 GPUs with DDP):

```bash
python train.py

# Resume from the last checkpoint of an existing run
python train.py resume_dir=outputs/actionplan

# Override any config value from the command line, e.g. single GPU
python train.py trainer.devices=1
```

Checkpoints and logs are written to `outputs/actionplan/` (the frozen config `config.json` there is what `generate.py` and `eval.py` read).

### Logging with Weights & Biases

Metrics are always written to a CSV logger under `outputs/actionplan/logs/`. W&B logging is set up but disabled by default; to enable it, log in once and pass `wandb_mode=online`:

```bash
wandb login          # once, or set WANDB_API_KEY
python train.py wandb_mode=online wandb_project=actionplan

# No internet on the training node? Log offline and sync later:
python train.py wandb_mode=offline
wandb sync outputs/actionplan/wandb/latest-run
```

Besides scalar metrics, the sample logger callback uploads rendered sample videos and generated-vs-ground-truth action plan text comparisons to W&B at every validation round (every 1000 epochs by default).

---

## Evaluation

Reproduces the ActionPlan numbers of the paper (Tables 1 and 4) on the HumanML3D-272 test set. Metrics: FID, R-Precision (Top1/2/3), Matching Score, and Diversity.

Requires the 272-dim test data:

```bash
python prepare/download_streamer272_data.py --skip_latents
```

Run the evaluation (paper settings: guidance 5.5, seed 123, ~20 replications):

```bash
# ActionPlan-Offline (Table 1)
python eval.py --sampler offline --replication_times 20

# ActionPlan-Streaming (Table 1)
python eval.py --sampler streaming --replication_times 20

# Sampling-strategy ablation (Table 4): parallel / offline_s5 / s10 / s15 / s25
python eval.py --sampler parallel --replication_times 20

# Everything in one go, generation parallelized over GPUs 1..3
python eval.py --sampler all --replication_times 20 --num_gpus 4

# Latency benchmark (Table 2)
python benchmark_latency.py --num_runs 20
```

All samplers share the same checkpoint; only the sampling schedule differs:

| `--sampler` | Paper entry |
| --- | --- |
| `offline` | ActionPlan-Offline (random pyramid, K=2 overlap) |
| `streaming` | ActionPlan-Streaming (text + first latent together, raster order) |
| `parallel` | Fully overlap (Parallel): all latents denoised jointly |
| `offline_s5/s10/s15` | K-step non-overlap rows |
| `offline_s25` | Fully non-overlap (Random): one latent at a time |

Results are printed as mean ± 95% CI and saved as JSON to `outputs/actionplan/eval_results/`.

---

## ✍️ Citation
If you find our work or code useful for your research, please consider citing:

```bibtex
@article{nazarenus2026actionplan,
  title   = {{ActionPlan}: Future-Aware Streaming Motion Synthesis via Frame-Level Action Planning},
  author  = {Nazarenus, Eric and Li, Chuqiao and He, Yannan and Xie, Xianghui and Lenssen, Jan Eric and Pons-Moll, Gerard},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## 👀 You Might Also Like

Also check out our CVPR 2026 paper 🧟 **[FrankenMotion](https://coral79.github.io/frankenmotion/)** — part-level human motion generation and composition with fine-grained spatial and temporal control.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Coral79/ActionPlan-Code&type=Date)](https://star-history.com/#Coral79/ActionPlan-Code&Date)
