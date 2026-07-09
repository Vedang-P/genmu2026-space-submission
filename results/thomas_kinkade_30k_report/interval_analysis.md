# Thomas Kinkade interval analysis

This supplement shows the pre-resized TorchMetrics cumulative FID at every 1,000 images.
The independently validated canonical native-preprocessing torch-fidelity value is reported separately at 30k.
CLIP, style, LPIPS, DINO, and ResNet metrics were defined on the 500 matched prompt pairs,
so they are shown as final paired metrics rather than fabricated 1k histories.

## Artifacts

- `fid_every_1k.csv`, `.png`, `.pdf`: cumulative FID and 1k-to-1k change.
- `all_metrics_dashboard.png`, `.pdf`: FID, ΔFID, final paired metrics, and training dynamics.
- `visual_examples.png`, `.pdf`: actual matched vanilla/SPACE generations.
- `interval_analysis.json`: machine-readable values and provenance notes.
