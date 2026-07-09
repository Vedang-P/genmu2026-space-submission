# Experiment Plan

## Objective

Train and evaluate SPACE on Thomas Kinkade without changing the model or evaluation architecture, using 50 target prompts with 10 samples per prompt and a 30,000-image COCO FID comparison against vanilla Stable Diffusion v1.4.

## Dataset

- Target evaluation: a dedicated 50-row Thomas Kinkade prompt CSV, extending the 20 existing prompts with 30 diverse, artist-relevant scenes. Each prompt keeps a fixed evaluation seed and produces 10 samples for 500 matched outputs per model.
- General preservation and FID: the repository's existing `data/coco_30k.csv`, using all 30,000 prompts and fixed seeds.
- FID reference: the exact official MS COCO 2014 validation images selected by `data/coco_30k.csv`. This is the standard FID protocol used by the ESD paper and avoids regenerating a redundant vanilla image set.

## Model Architecture

Keep the existing Stable Diffusion v1.4 + SPACE implementation unchanged: saliency-selected cross-attention LoRA adapters, dual anchors, style-subspace suppression, trajectory-band supervision, and preservation replay. Only experiment orchestration, prompt data, progress persistence, metric reporting, and W&B instrumentation will be extended.

## Training Protocol

- Artist: Thomas Kinkade.
- Existing SPACE defaults: 300 stage-1 steps, 100 stage-2 steps, learning rate `1e-4`, LoRA rank 8, trajectory bands `0.2,0.5,0.8`, and the current loss weights.
- Save compact training state/metrics periodically under the persistent RunPod `/workspace` volume and write the final checkpoint to the existing `space-models/sd` layout.
- Generate images idempotently: every completed PNG is immediately durable and reruns skip valid existing outputs.
- Use a stable W&B run ID with `resume="allow"`; log training losses, generation counts/rates, cumulative FID milestones, final metrics, plots, and compact result artifacts.

## Baselines

- Vanilla Stable Diffusion v1.4 for paired target-prompt outputs. Its published COCO-30k FID of 14.50 from the ESD paper is reported as context, while SPACE FID is computed against official COCO images.
- The existing Thomas Kinkade ESD-x outputs remain available for contextual comparison, but this extension does not retrain or restructure baseline methods.

## Evaluation Metrics

- Primary: standard FID between Thomas Kinkade SPACE generations and official COCO images, reported cumulatively at 1k, 5k, 10k, 15k, 20k, 25k, and 30k images.
- Existing paired metrics on 500 matched prompt outputs: CLIP semantic similarity/drop, style score/drop and target rate, LPIPS, ResNet agreement, DINO similarity where available, and KID.
- Visuals: FID-versus-sample-count curve, periodic metric table, existing comparison bars, and a representative paired-image grid.

## Persistence And Repository Artifacts

- Persist generated images, logs, checkpoints, W&B local state, manifests, and progress JSONL under `/workspace` on RunPod.
- Commit only reproducible inputs and compact outputs: the 50-prompt CSV, run configuration/provenance, metric CSV/JSON, summary Markdown, plots, and selected comparison grids. Do not commit the 60,000 COCO FID PNGs or bulk checkpoints.
- Provide an explicit finalization command that validates expected counts and stages the compact result allowlist for a normal Git commit; it will not silently push or commit partial results.

## Ablations

No new ablation or architecture change is introduced in this run. The cumulative FID milestones measure estimator stabilization with sample count, not a model ablation.
