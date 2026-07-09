# SPACE: Style-Preserving Attribute-Constrained Erasure

SPACE is a preservation-first method for artist/style unlearning in Stable Diffusion v1.4. It edits the model so that prompts containing a target artist cue no longer reproduce that recognizable style, while content, general image quality, and nearby non-target styles stay as close to the base model as possible. Submitted to the [Genμ 2.0 Challenge](https://iab-iitj.github.io/genmu2026/evaluation.html) (Track 3: Visual Concept Unlearning).

For a prompt like `Bedroom in Arles by Vincent van Gogh`, the edited model should still produce a coherent bedroom — not a collapsed or unrelated image, and not another named artist's style. SPACE targets **neutralized style suppression**, not artist replacement and not indiscriminate quality degradation.

## Submission target: Van Gogh

The submitted checkpoint erases **Vincent van Gogh** from CompVis Stable Diffusion v1.4. Validated on 50 target prompts (10 samples each, 500 matched pairs) plus a full 30,000-image FID run against the official MS COCO 2014 validation set.

| Metric | SPACE (Van Gogh) | Reference |
| --- | --- | --- |
| FID (30k, `torch-fidelity`, canonical) | **14.57** | Vanilla SD v1.4 (ESD paper): 14.50 |
| CLIP image similarity (edited vs. vanilla) | 0.798 | 1.0 = identical |
| Style target rate | 0.324 | fraction of forget-set samples still hitting the target style |
| Style drop (CLIP style score) | 0.045 | vanilla style score 0.298 → edited 0.252 |
| LPIPS (edited vs. vanilla) | 0.569 | perceptual distance |
| DINO similarity (edited vs. vanilla) | 0.622 | structural/content similarity |

FID lands within ~0.1 of the unedited base model while the target style score drops — general image quality is preserved at COCO scale while the target concept is suppressed. Full protocol, cumulative FID-vs-sample-count curve, training curves, and a metric-validation audit are in [`results/van_gogh_30k_report/`](results/van_gogh_30k_report/) (start with [`summary.md`](results/van_gogh_30k_report/summary.md)).

- **Model weights:** [Google Drive](https://drive.google.com/drive/folders/1L07gdJJQc1Tmf4sj4HkkAuvDdIpWDPIp)
- **Retain-set samples (COCO, SPACE-generated):** [HF: `van_gogh_30k/fid/space`](https://huggingface.co/datasets/vedangfake/space-erasing-artifacts/tree/main/van_gogh_30k/fid/space)
- **Forget-set samples (Van Gogh prompts, vanilla vs. SPACE):** [HF: `van_gogh_30k/report/visual_examples.png`](https://huggingface.co/datasets/vedangfake/space-erasing-artifacts/blob/main/van_gogh_30k/report/visual_examples.png)

## Additional validated result: Thomas Kinkade

SPACE was validated with the same protocol on a second, independent concept:

| Metric | SPACE (Thomas Kinkade) | Reference |
| --- | --- | --- |
| FID (30k, canonical) | 14.58 | Vanilla SD v1.4: 14.50 |
| CLIP image similarity | 0.755 | 1.0 = identical |
| Style target rate | 0.484 | fraction still hitting target style |
| Style drop | 0.064 | vanilla style score 0.298 → edited 0.234 |
| LPIPS | 0.578 | perceptual distance |
| DINO similarity | 0.471 | structural/content similarity |

Details in [`results/thomas_kinkade_30k_report/`](results/thomas_kinkade_30k_report/). Generated samples (forget + retain) for both concepts are on Hugging Face: **[vedangfake/space-erasing-artifacts](https://huggingface.co/datasets/vedangfake/space-erasing-artifacts)** (`van_gogh_30k/`, `thomas_kinkade_30k/`).

## How SPACE works

For each target prompt, SPACE builds four prompt views and edits the model relative to its own frozen predictions — there is no separate teacher network.

| Symbol | Meaning | Example for `Bedroom in Arles by Vincent van Gogh` |
| --- | --- | --- |
| `p_t` | target prompt with artist/style | `Bedroom in Arles by Vincent van Gogh` |
| `p_c` | content-only prompt | `Bedroom in Arles` |
| `p_a` | neutral art anchor | `a high quality painting of Bedroom in Arles` |
| `p_g` | generic image anchor | `a high quality image of Bedroom in Arles` |

```text
neutral_anchor  = (1 - alpha_art) * eps_base(x_t, p_c) + alpha_art * eps_base(x_t, p_a)
style_direction = eps_base(x_t, p_t) - eps_base(x_t, p_c)
teacher         = neutral_anchor - erase_scale * style_direction
```

The teacher anchors the target prompt toward neutral, content-preserving output *and* explicitly subtracts the estimated target-style residual, rather than just imitating a generic replacement distribution. The base Stable Diffusion U-Net is frozen; SPACE trains only saliency-selected LoRA adapters on cross-attention projections, supervised across multiple denoising trajectory bands (`0.2, 0.5, 0.8`) with a six-term loss:

```text
L_total = L_erase + λ_style·L_style + λ_content·L_content + λ_image·L_image
        + λ_preserve·L_preserve + λ_vulnerable·L_vulnerable + λ_lora·L_lora
```

- **`L_erase`** — moves the target-prompt prediction toward the SPACE teacher.
- **`L_style`** — suppresses a learned low-rank *style subspace* (not just one residual vector), built via SVD over several target-prompt variants.
- **`L_content` / `L_image`** — keep content-only and generic-image predictions close to the frozen base model.
- **`L_preserve`** — replays general prompts from `data/coco_30k.csv` on a separately sampled latent.
- **`L_vulnerable`** — mines and preserves nearby artist styles (via CLIP text similarity) most at risk of collateral damage.
- **`L_lora`** — regularizes adapter magnitude to keep the edit localized.

What's novel relative to the baselines it's benchmarked against:

| Method | Erasure mechanism | Preservation mechanism | Limitation SPACE targets |
| --- | --- | --- | --- |
| ESD-x | gradient edit moving away from target concept predictions | narrow parameter scope | erasure strength can trade off hard against content/quality |
| UCE | closed-form linear edit for erased + preserved concepts | explicit preserve concept list | less trajectory-aware, less content/style factorized per prompt |
| Concept Ablation | maps target concept toward an anchor distribution | anchor preserves visual plausibility | can under-erase and damage nearby concepts |
| **SPACE** | dual-anchor teacher + explicit negative style-direction subtraction | content anchor, image anchor, general-prompt replay, vulnerable-style replay, LoRA regularization | combines CA-style visual plausibility with stronger, more controlled erasure |

Full notation, the saliency-ranking procedure, the two-stage training curriculum, and implementation-accuracy notes are in [`docs/METHOD.md`](docs/METHOD.md).

## Repository structure

```text
training/    Training entrypoints (space_sd.py, esd_sd.py) and run scripts,
             including the Thomas Kinkade 30k submission-pipeline example.
inference/   Given a checkpoint (or none, for vanilla SD), generate images.
evaluation/  CLIP/LPIPS/FID/KID metrics, comparison grids, FID-vs-sample-count
             analysis, report generation.
core/        Shared library code: the SPACE trainer, the ESD trainer, LoRA
             checkpoint I/O, and Stable Diffusion call utilities.
baselines/   Official UCE and Concept Ablation harnesses (setup, training,
             generation wrappers around the upstream repos).
data/        Prompt CSVs: target-artist prompts and the COCO-30k preserve/FID set.
environment/ RunPod bootstrap and ESD-x baseline weight download.
docs/        Full method write-up, research objective, Thomas Kinkade RunPod
             run guide, experiment plan.
results/     Compact, checked-in metrics/plots/provenance for the Van Gogh
             submission and the Thomas Kinkade validation run. Bulk generated
             images and checkpoints are not committed — see "Weights and
             generated samples" below.
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # needed to pull CompVis/stable-diffusion-v1-4
```

All scripts assume they're run from the repository root and resolve the base model as `CompVis/stable-diffusion-v1-4` via `diffusers`.

## Usage

### Train SPACE on Van Gogh (submission target)

```bash
ERASE_CONCEPT="Vincent van Gogh" \
TARGET_PROMPTS_PATH="data/vangogh_prompts.csv" \
TARGET_ARTIST_FILTER="Vincent van Gogh" \
EXP_NAME="Van_Gogh" \
bash training/run_space_training.sh
```

Checkpoint is written to `space-models/sd/space-Van_Gogh.safetensors`. Key knobs (`ERASE_SCALE`, `ALPHA_ART`, `TRAJECTORY_BANDS`, `LORA_RANK`, stage lengths, loss weights) are documented in [`docs/METHOD.md`](docs/METHOD.md) and exposed as env vars in [`training/run_space_training.sh`](training/run_space_training.sh). A 10-step sanity check: `SPACE_DEBUG_STEPS=10 bash training/run_space_training.sh`.

The validated 30k-image COCO-FID protocol (the numbers reported above) chains this training step with `inference/generate_images.py`, `evaluation/generate_fid_samples.py`, and `evaluation/evaluate.py` over 50 target prompts × 10 samples and 30,000 COCO prompts. [`training/run_thomas_kinkade_30k.sh`](training/run_thomas_kinkade_30k.sh) is a concrete, artist-specific worked example of that exact pipeline shape — swap in `data/vangogh_prompts.csv` and the Van Gogh checkpoint/filter to reproduce the Van Gogh run.

### Reproduce the Thomas Kinkade 30k pipeline (second validated concept)

Trains SPACE on Thomas Kinkade, generates the 500 matched prompt-pair images, generates 30,000 SPACE images, downloads the official COCO reference, and computes cumulative FID through 30k:

```bash
WANDB_PROJECT=space-thomas-kinkade FID_BATCH_SIZE=8 \
bash training/run_thomas_kinkade_30k.sh
```

See [`docs/THOMAS_KINKADE_30K_RUNPOD.md`](docs/THOMAS_KINKADE_30K_RUNPOD.md) for pod sizing, persistent storage, and recovery notes.

### Generate images from a trained checkpoint (inference)

```bash
python inference/generate_images.py \
  --base_model CompVis/stable-diffusion-v1-4 \
  --esd_path space-models/sd/space-Van_Gogh.safetensors \
  --prompts_path data/vangogh_prompts.csv \
  --save_path results/space/space-Van_Gogh \
  --num_samples 10 --num_inference_steps 20 --guidance_scale 7.5
```

Omit `--esd_path` to generate vanilla (unedited) SD v1.4 images for the same prompts. `inference/run_baseline.sh` and `inference/run_space_images.sh` wrap this for the standard benchmark layout.

### Evaluate

```bash
python evaluation/evaluate.py --only-artist "Van Gogh" --method-keys space
python evaluation/make_comparison_grids.py --only-artist "Van Gogh" --methods SPACE
```

Computes CLIP erasure/preservation scores, LPIPS, FID/KID, ResNet agreement, and DINO similarity; writes `results/evaluation/metrics.csv` and a side-by-side comparison grid.

## Baselines

[`baselines/`](baselines/) contains wrappers around the **official** UCE and Concept Ablation implementations (cloned to pinned commits, not vendored) plus the ESD-x replication path (`training/esd_sd.py`, `inference/generate_esd_old.py`), used during method development to benchmark SPACE against prior work on a shared Van Gogh prompt set. See [`baselines/README.md`](baselines/README.md) for setup and reproduction.

## Weights and generated samples

Checkpoints and full generated-image galleries are hosted externally rather than committed to git:

- **Model weights (SPACE LoRA checkpoint, Van Gogh):** [Google Drive](https://drive.google.com/drive/folders/1L07gdJJQc1Tmf4sj4HkkAuvDdIpWDPIp)
- **Generated samples — retain set (COCO, SPACE-generated):** [HF: `van_gogh_30k/fid/space`](https://huggingface.co/datasets/vedangfake/space-erasing-artifacts/tree/main/van_gogh_30k/fid/space)
- **Generated samples — forget set (Van Gogh prompts, vanilla vs. SPACE):** [HF: `van_gogh_30k/report/visual_examples.png`](https://huggingface.co/datasets/vedangfake/space-erasing-artifacts/blob/main/van_gogh_30k/report/visual_examples.png)
- **Full dataset (both concepts, all splits):** [huggingface.co/datasets/vedangfake/space-erasing-artifacts](https://huggingface.co/datasets/vedangfake/space-erasing-artifacts)

`results/evaluation/` and `results/provenance/` carry small, checked-in metrics/config/plots. `results/van_gogh_30k_report/` and `results/thomas_kinkade_30k_report/` are the compact, validated report bundles for each concept (metrics, FID-vs-sample-count curve, training curves, a metric-validation audit, and a representative visual comparison — the full 500-pair comparison grids are on Hugging Face instead of in git).

## License and attribution

Code is MIT-licensed — see [`LICENSE`](LICENSE). This repository builds on the official ESD implementation ([erasing.baulab.info](https://erasing.baulab.info/)) and benchmarks against UCE ([unified.baulab.info](https://unified.baulab.info/)) and Concept Ablation ([cs.cmu.edu/~concept-ablation](https://www.cs.cmu.edu/~concept-ablation/)); official baseline source is not vendored and is cloned locally by [`baselines/setup_official_repos.sh`](baselines/setup_official_repos.sh).

SPACE targets output-level artist/style suppression under this evaluation harness — this is not a claim of complete removal of target-concept knowledge from model weights.
