# Thomas Kinkade SPACE 30k Evaluation

## Protocol

- Model: SPACE edit of CompVis Stable Diffusion v1.4
- Target prompts: 50
- Samples per target prompt: 10
- Matched prompt pairs: 500
- FID: 30,000 SPACE generations against their official MS COCO 2014 validation images
- Published SD v1.4 COCO-30k FID (ESD paper): 14.50, reported separately as context
- FID milestones: 1k, 5k, 10k, 15k, 20k, 25k, 30k

## Final metrics

- Canonical torch-fidelity FID (30k): 14.578488
- Prior pre-resized TorchMetrics FID (noncanonical): 14.089452
- Published vanilla SD v1.4 FID: 14.50 (context only; generation settings are not proven identical)
- CLIP image similarity: 0.754964
- Style target rate: 0.484000
- Style drop: 0.064447
- LPIPS: 0.578449
- DINO similarity: 0.470875

## Artifacts

- `metrics.csv`: final paired metrics plus 30k FID
- `fid_progress.csv`: cumulative FID at each milestone
- `fid_progress.png`: FID stabilization curve
- `prompt_pair_grid.png`: representative vanilla/SPACE pairs
- `training_metrics.csv` and `training_metrics.png`: periodic training metrics

Bulk generated PNGs and model checkpoints are intentionally excluded from Git.

- `metric_validation.md`: independent implementation and input audit
