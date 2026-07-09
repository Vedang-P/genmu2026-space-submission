#!/usr/bin/env python3
"""Build compact, publication-ready interval charts and visual examples."""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "navy": "#17324d",
    "blue": "#2878b5",
    "orange": "#e07a35",
    "green": "#3a8f6b",
    "gray": "#6b7280",
    "light": "#eef3f7",
}


def save_figure(fig, output_base: Path):
    fig.savefig(output_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fid_figures(fid: pd.DataFrame, report: Path, canonical_fid=None):
    x = fid["n_images"].to_numpy() / 1000
    y = fid["fid"].to_numpy()
    delta = fid["fid"].diff().to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(x, y, color=COLORS["blue"], marker="o", markersize=4, linewidth=2)
    ax1.axhline(14.50, color=COLORS["orange"], linestyle="--", linewidth=1.5, label="Published SD v1.4: 14.50")
    ax1.set_ylabel("FID vs. official COCO (lower is better)")
    if canonical_fid is not None:
        ax1.axhline(canonical_fid, color=COLORS["green"], linestyle=":", linewidth=1.5, label=f"Canonical torch-fidelity 30k: {canonical_fid:.3f}")
    ax1.set_title("Thomas Kinkade SPACE — pre-resized TorchMetrics FID every 1,000 images")
    ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    bar_colors = [COLORS["green"] if value <= 0 else COLORS["orange"] for value in np.nan_to_num(delta)]
    ax2.bar(x[1:], delta[1:], width=0.72, color=bar_colors[1:])
    ax2.axhline(0, color="#222222", linewidth=0.8)
    ax2.set_xlabel("Cumulative images (thousands)")
    ax2.set_ylabel("Change from prior 1k")
    ax2.set_xticks(x)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, report / "fid_every_1k")


def dashboard(fid: pd.DataFrame, metrics: pd.Series, training: pd.DataFrame, audit: dict, report: Path, canonical_fid=None):
    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    x = fid["n_images"] / 1000
    ax.plot(x, fid["fid"], color=COLORS["blue"], marker="o", linewidth=2)
    ax.axhline(14.50, color=COLORS["orange"], linestyle="--", linewidth=1.3)
    if canonical_fid is not None:
        ax.axhline(canonical_fid, color=COLORS["green"], linestyle=":", linewidth=1.3)
    ax.set(title="Pre-resized TorchMetrics FID stabilization", xlabel="Images (thousands)", ylabel="FID")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    delta = fid["fid"].diff()
    ax.bar(x.iloc[1:], delta.iloc[1:], width=0.72, color=COLORS["green"])
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set(title="FID change every 1,000 images", xlabel="Images (thousands)", ylabel="Δ FID")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    final_names = ["CLIP image sim.", "Style target rate", "Style drop", "LPIPS", "DINO sim.", "Top-1 agreement", "Top-5 overlap"]
    final_values = [
        metrics["clip_image_similarity"], metrics["style_target_rate"], metrics["style_drop"],
        metrics["lpips"], metrics["dino_similarity"], metrics["resnet_top1_agreement"], metrics["resnet_top5_overlap"],
    ]
    order = np.arange(len(final_names))
    ax.barh(order, final_values, color=[COLORS["blue"], COLORS["orange"], COLORS["orange"], COLORS["gray"], COLORS["green"], COLORS["navy"], COLORS["navy"]])
    ax.set_yticks(order, final_names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Final paired metrics (500 matched pairs)")
    for i, value in enumerate(final_values):
        ax.text(value + 0.015, i, f"{value:.3f}", va="center", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1])
    if "step" not in training.columns:
        training = training.copy()
        training["step"] = np.arange(1, len(training) + 1)
    candidates = [("loss", "Total loss"), ("erase_loss", "Erase loss"), ("content_loss", "Content loss"), ("preserve_loss", "Preserve loss")]
    for key, label in candidates:
        if key in training.columns:
            ax.plot(training["step"], training[key], label=label, linewidth=1.3)
    ax.set(title="Training dynamics", xlabel="Training step", ylabel="Loss")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Thomas Kinkade SPACE experiment dashboard  |  Canonical FID {float(metrics['fid']):.3f}  |  "
        f"Black images {audit['exact_black_images']}/{audit['space_images_scanned']} ({100*audit['exact_black_fraction']:.2f}%)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    save_figure(fig, report / "all_metrics_dashboard")


def image_gallery(pair_root: Path, prompts_path: Path, report: Path, max_pairs: int = 12):
    baseline_dir = pair_root / "baseline"
    space_dir = pair_root / "space"
    common = sorted({p.name for p in baseline_dir.glob("*.png")} & {p.name for p in space_dir.glob("*.png")})
    if not common:
        raise RuntimeError("No matched baseline/SPACE prompt images found")
    indexes = np.linspace(0, len(common) - 1, min(max_pairs, len(common)), dtype=int)
    selected = [common[index] for index in indexes]
    prompts = pd.read_csv(prompts_path)
    prompt_by_case = {int(row.case_number): str(row.prompt) for _, row in prompts.iterrows()}

    thumb = 256
    label_h = 72
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (cols * thumb * 2, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for slot, filename in enumerate(selected):
        row, col = divmod(slot, cols)
        case = int(filename.split("_")[0])
        prompt = prompt_by_case.get(case, f"Case {case}")
        left = col * thumb * 2
        top = row * (thumb + label_h)
        for offset, directory in [(0, baseline_dir), (thumb, space_dir)]:
            image = Image.open(directory / filename).convert("RGB")
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb, thumb), "#eeeeee")
            tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
            canvas.paste(tile, (left + offset, top + label_h))
        draw.text((left + 6, top + 4), "VANILLA", fill=COLORS["navy"], font=font)
        draw.text((left + thumb + 6, top + 4), "SPACE", fill=COLORS["orange"], font=font)
        wrapped = textwrap.wrap(prompt, width=64)[:3]
        draw.multiline_text((left + 6, top + 20), "\n".join(wrapped), fill="#222222", font=font, spacing=2)
    canvas.save(report / "visual_examples.png", optimize=True)
    canvas.save(report / "visual_examples.pdf", "PDF", resolution=150)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--fid_csv", required=True)
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--prompts_path", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    report = run_root / "report"
    report.mkdir(parents=True, exist_ok=True)
    fid = pd.read_csv(args.fid_csv)
    if fid["n_images"].tolist() != list(range(1000, 30001, 1000)):
        raise RuntimeError("Expected honest cumulative FID milestones at every 1,000 images")
    fid["fid_change_1k"] = fid["fid"].diff()
    fid.to_csv(report / "fid_every_1k.csv", index=False)
    metrics = pd.read_csv(args.metrics_csv).iloc[0]
    training = pd.read_csv(report / "training_metrics.csv")
    audit = json.loads((report / "black_image_audit.json").read_text())
    validation_path = report / "metric_validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    canonical_fid = validation["fid"]["canonical_torch_fidelity_native_preprocessing"] if validation else None

    fid_figures(fid, report, canonical_fid)
    dashboard(fid, metrics, training, audit, report, canonical_fid)
    selected = image_gallery(run_root / "prompt_pairs", Path(args.prompts_path), report)

    stats = {
        "canonical_torch_fidelity_fid_30k": canonical_fid,
        "pre_resized_torchmetrics_fid_30k": float(fid.iloc[-1]["fid"]),
        "published_sd14_fid_context": 14.5,
        "fid_every_1k": fid.to_dict(orient="records"),
        "paired_metrics": {key: (None if pd.isna(value) else value) for key, value in metrics.to_dict().items()},
        "black_image_audit": audit,
        "visual_example_files": selected,
        "methodology_note": "The 1k curve uses the documented pre-resized TorchMetrics implementation. Canonical native-preprocessing torch-fidelity is reported separately at 30k. Paired metrics are final values over 500 matched pairs and are not presented as 1k interval series.",
    }
    (report / "interval_analysis.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")

    lines = [
        "# Thomas Kinkade interval analysis",
        "",
        "This supplement shows the pre-resized TorchMetrics cumulative FID at every 1,000 images.",
        "The independently validated canonical native-preprocessing torch-fidelity value is reported separately at 30k.",
        "CLIP, style, LPIPS, DINO, and ResNet metrics were defined on the 500 matched prompt pairs,",
        "so they are shown as final paired metrics rather than fabricated 1k histories.",
        "",
        "## Artifacts",
        "",
        "- `fid_every_1k.csv`, `.png`, `.pdf`: cumulative FID and 1k-to-1k change.",
        "- `all_metrics_dashboard.png`, `.pdf`: FID, ΔFID, final paired metrics, and training dynamics.",
        "- `visual_examples.png`, `.pdf`: actual matched vanilla/SPACE generations.",
        "- `interval_analysis.json`: machine-readable values and provenance notes.",
    ]
    (report / "interval_analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
