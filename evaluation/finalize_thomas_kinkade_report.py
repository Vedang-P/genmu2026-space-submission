#!/usr/bin/env python3
"""Build the compact, commit-safe report bundle for the Thomas Kinkade run."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--paired_metrics", required=True)
    parser.add_argument("--comparison_grid", required=True)
    args = parser.parse_args()

    root = Path(args.run_root)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    fid_summary = json.loads((root / "fid_analysis/fid_summary.json").read_text())
    fid_progress = pd.read_csv(root / "fid_analysis/fid_progress.csv")
    paired = pd.read_csv(args.paired_metrics)
    paired = paired[(paired["artist"] == "Thomas Kinkade") & (paired["method"] == "SPACE")].copy()
    if len(paired) != 1:
        raise RuntimeError(f"Expected one Thomas Kinkade / SPACE metrics row, found {len(paired)}")
    paired["fid"] = float(fid_summary["final_fid"])
    paired.to_csv(report / "metrics.csv", index=False)
    fid_progress.to_csv(report / "fid_progress.csv", index=False)
    shutil.copy2(root / "fid_analysis/fid_progress.png", report / "fid_progress.png")
    shutil.copy2(args.comparison_grid, report / "prompt_pair_grid.png")
    training_jsonl = root / "progress/training/training_metrics.jsonl"
    if training_jsonl.exists():
        training = pd.read_json(training_jsonl, lines=True).drop_duplicates("step", keep="last").sort_values("step")
        training.to_csv(report / "training_metrics.csv", index=False)
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(training["step"], training["loss"], label="total")
        axes[0].plot(training["step"], training["erase_loss"], label="erase", alpha=0.8)
        axes[0].plot(training["step"], training["style_loss"], label="style", alpha=0.8)
        axes[0].set_ylabel("Loss")
        axes[0].legend(ncol=3)
        axes[0].grid(alpha=0.2)
        axes[1].plot(training["step"], training["content_loss"], label="content")
        axes[1].plot(training["step"], training["image_loss"], label="image")
        axes[1].plot(training["step"], training["preserve_loss"], label="preserve")
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Preservation loss")
        axes[1].legend(ncol=3)
        axes[1].grid(alpha=0.2)
        fig.suptitle("Thomas Kinkade SPACE training metrics")
        fig.tight_layout()
        fig.savefig(report / "training_metrics.png", dpi=180)
        plt.close(fig)

    row = paired.iloc[0]
    lines = [
        "# Thomas Kinkade SPACE 30k Evaluation",
        "",
        "## Protocol",
        "",
        "- Model: SPACE edit of CompVis Stable Diffusion v1.4",
        "- Target prompts: 50",
        "- Samples per target prompt: 10",
        "- Matched prompt pairs: 500",
        "- FID: 30,000 SPACE generations against their official MS COCO 2014 validation images",
        "- Published SD v1.4 COCO-30k FID (ESD paper): 14.50, reported separately as context",
        "- FID milestones: 1k, 5k, 10k, 15k, 20k, 25k, 30k",
        "",
        "## Final metrics",
        "",
        f'- FID (30k): {float(row["fid"]):.6f}',
        '- Published vanilla SD v1.4 FID: 14.50',
        f'- CLIP image similarity: {float(row["clip_image_similarity"]):.6f}',
        f'- Style target rate: {float(row["style_target_rate"]):.6f}',
        f'- Style drop: {float(row["style_drop"]):.6f}',
        f'- LPIPS: {float(row["lpips"]):.6f}',
        f'- DINO similarity: {float(row["dino_similarity"]):.6f}',
        "",
        "## Artifacts",
        "",
        "- `metrics.csv`: final paired metrics plus 30k FID",
        "- `fid_progress.csv`: cumulative FID at each milestone",
        "- `fid_progress.png`: FID stabilization curve",
        "- `prompt_pair_grid.png`: representative vanilla/SPACE pairs",
        "- `training_metrics.csv` and `training_metrics.png`: periodic training metrics",
        "",
        "Bulk generated PNGs and model checkpoints are intentionally excluded from Git.",
    ]
    (report / "summary.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "status": "completed",
        "prompt_count": 50,
        "samples_per_prompt": 10,
        "paired_images_per_model": 500,
        "fid_images_per_model": 30000,
        "final_fid": float(fid_summary["final_fid"]),
        "report_files": sorted(path.name for path in report.iterdir()),
    }
    (report / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print((report / "summary.md").read_text())


if __name__ == "__main__":
    main()
