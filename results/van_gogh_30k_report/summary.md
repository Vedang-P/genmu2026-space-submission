# Van Gogh SPACE 30k Evaluation

## Protocol

- Model: SPACE edit of CompVis Stable Diffusion v1.4
- Target prompts: 50
- Samples per target prompt: 10
- Matched prompt pairs: 500
- FID: 30,000 SPACE generations against their official MS COCO 2014 validation images (same reference set used for Thomas Kinkade)
- Published SD v1.4 COCO-30k FID (ESD paper, clean-fid implementation): 14.50, reported separately as context
- FID milestones: 1k, 5k, 10k, 15k, 20k, 25k, 30k

## Final metrics

- Canonical torch-fidelity FID (30k): 14.567865
- Published vanilla SD v1.4 FID: 14.50
- CLIP image similarity: 0.797599
- Style target rate: 0.324000
- Style drop: 0.045477
- LPIPS: 0.568588
- DINO similarity: 0.622103

## Artifacts

- `metrics.csv`: final paired metrics plus canonical 30k FID
- `fid_progress.csv`, `fid_every_1k.png/.pdf`: cumulative pre-resized-TorchMetrics FID at each 1k milestone
- `metric_validation.json/.md`: independent canonical-FID and input audit
- `all_metrics_dashboard.png`: FID, paired metrics, and training dynamics in one view
- `prompt_pair_grid.png`, `visual_examples.png/.pdf`: representative vanilla/SPACE pairs
- `training_metrics.csv/.png`: periodic training metrics

Bulk generated PNGs and model checkpoints are intentionally excluded from Git.
