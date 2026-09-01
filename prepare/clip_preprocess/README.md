# Frame-level CLIP preprocessing

This folder rebuilds the per-latent CLIP embeddings that ActionPlan concatenates with TAE motion latents.

**We do not provide BABEL annotations, BABEL text, or any JSON derived from BABEL.**
Get BABEL yourself from [https://babel.is.tue.mpg.de/](https://babel.is.tue.mpg.de/) and follow their license (non-commercial academic use). Convert those labels into the JSON layout below before running anything here.

You do **not** need this pipeline to train or run ActionPlan. `python prepare/download_dependencies.py` already ships the finished 32-d files in `datasets/motions/t2m_latent_frame_text_aligned/`.

Step 1 writes `datasets/embeddings/temp_clip/clip_enc_latent_aligned/`, which `models/clip_autoencoder/2_train.py` uses as `DEFAULT_DATA_DIR`.

---

## What you must provide

One JSON file per motion, named with the HumanML3D id:

```
datasets/embeddings/temp_clip/babel_frame_labels/000002.json
```

Exact schema (`clip_encoder.py` reads only this):

```json
{
  "frame_label": [
    ["walk"],
    ["walk", "forward"],
    ["stand"]
  ]
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| filename stem | yes | HumanML3D motion id, e.g. `000002.json` → `000002.npy` |
| `frame_label` | yes | List, one entry per **20 fps** frame |
| `frame_label[t]` | yes | Non-empty list of BABEL action-label strings for that frame |

- `len(frame_label)` must be the number of 20 fps frames of that sequence.
- Several labels on one frame are averaged in CLIP space (`--multi avg`, default). `--multi single` keeps only the first label.
- No other JSON keys are used. Do not point this script at raw BABEL release files; they are a different format.

HumanML3D ids that have frame-level text in the paper split (after `download_dependencies.py`):

`datasets/annotations/humanml3d_actionplan_merged/splits/train_frame_level_subset.txt`

---

## Pipeline (from repo root)

Also required, not from BABEL:

- 16-d TAE motion latents: `python prepare/download_streamer272_data.py` → `datasets/motions/t2m_latents/` (`[L, 16]`)
- Label bank (already in the deps zip): `datasets/embeddings/raw/clip_embeddings.tsv` and `label_to_id.json` (CLIP vectors of short action names, **not** BABEL source text)

```bash
# 0. CLIP-encode your JSON labels → [T_20, 512] at 20 fps
python prepare/clip_preprocess/clip_encoder.py \
  --annotation_dir datasets/embeddings/temp_clip/babel_frame_labels \
  --output_dir datasets/embeddings/temp_clip/clip_enc_single

# 1a. Resample 20 fps → 30 fps (nearest frame, no interpolation)
python prepare/clip_preprocess/0_clip_enc_single_to_30fps.py

# 1b. Average CLIP over the TAE window [4i-16, 4i+4] → [L, 512]
#     L is taken from datasets/motions/t2m_latents/<id>.npy
python prepare/clip_preprocess/1_align_clip_30fps_to_latent.py

# 2. Train 512→16 CLIP autoencoder on clip_enc_latent_aligned/
python models/clip_autoencoder/2_train.py \
  --train-split datasets/annotations/humanml3d_actionplan_merged/splits/train_frame_level_subset.txt \
  --val-split datasets/annotations/humanml3d_actionplan_merged/splits/val.txt

# 3. Encode all motions to 16-d
python models/clip_autoencoder/3_encode_all.py \
  --checkpoint outputs/clip_autoencoder/best.pt

# 4. Concat motion [16,L] + CLIP [16,L] → datasets/motions/t2m_latent_frame_text_aligned/[32, L]
python prepare/clip_preprocess/4_build_frame_text_aligned.py
```

A trained AE checkpoint is also shipped at `models/clip_autoencoder/best.pt`, so step 2 can be skipped if you only want to encode with the paper compressor.

---

## Outputs

| Path | Shape | Role |
| --- | --- | --- |
| `.../clip_enc_single/<id>.npy` | `[T_20, 512]` | CLIP per 20 fps frame |
| `.../clip_enc_single_30fps/<id>.npy` | `[T_30, 512]` | Same, 30 fps |
| `.../clip_enc_latent_aligned/<id>.npy` | `[L, 512]` | **`DEFAULT_DATA_DIR`** |
| `.../clip_enc_latent_aligned_ae16/<id>.npy` | `[L, 16]` | AE-compressed text latents |
| `datasets/motions/t2m_latent_frame_text_aligned/<id>.npy` | `[32, L]` | Training input (motion + text) |
