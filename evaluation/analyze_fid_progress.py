#!/usr/bin/env python3
"""Compute cumulative FID once, recording values at increasing sample milestones."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


FID_STATE_NAMES = (
    "real_features_sum",
    "real_features_cov_sum",
    "real_features_num_samples",
    "fake_features_sum",
    "fake_features_cov_sum",
    "fake_features_num_samples",
)


def metric_state(metric):
    return {name: getattr(metric, name).detach().cpu() for name in FID_STATE_NAMES}


def restore_metric_state(metric, state, device):
    for name in FID_STATE_NAMES:
        setattr(metric, name, state[name].to(device))


def image_tensor(path: Path) -> torch.Tensor:
    # COCO reference images have mixed native resolutions, while generated
    # images are square. Normalize both inputs before stacking a batch.
    image = Image.open(path).convert("RGB").resize((299, 299), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)


def resolve_pairs(prompts_path: str, baseline_dir: str, method_dir: str, total: int):
    df = pd.read_csv(prompts_path).head(total)
    pairs = []
    missing = []
    for _, row in df.iterrows():
        image_id = int(row.image_id)
        baseline_candidates = [
            Path(baseline_dir) / f"{image_id}.jpg",
            Path(baseline_dir) / f"{image_id}.png",
            Path(baseline_dir) / f"COCO_val2014_{image_id:012d}.jpg",
        ]
        baseline = next((path for path in baseline_candidates if path.exists()), baseline_candidates[0])
        method = Path(method_dir) / f"{image_id}.png"
        if not baseline.exists() or not method.exists():
            missing.append(str(image_id))
        else:
            pairs.append((baseline, method))
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Missing {len(missing)} matched FID image pairs; first missing: {preview}")
    if len(pairs) != total:
        raise RuntimeError(f"Resolved {len(pairs)} pairs; expected {total}")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--method_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--milestones", default="1000,5000,10000,15000,20000,25000,30000")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_id", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="disabled")
    args = parser.parse_args()

    milestones = sorted({int(value) for value in args.milestones.split(",") if value.strip()})
    if not milestones or milestones[0] < 2:
        raise ValueError("At least one FID milestone >= 2 is required")
    total = milestones[-1]
    pairs = resolve_pairs(args.prompts_path, args.baseline_dir, args.method_dir, total)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    rows = []
    processed = 0
    state_path = output / "fid_metric_state.pt"
    state_signature = {
        "baseline_dir": str(Path(args.baseline_dir).resolve()),
        "method_dir": str(Path(args.method_dir).resolve()),
        "total": total,
        "preprocessing": "rgb-resize-299-bilinear",
    }
    if state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False)
        if saved.get("signature") == state_signature:
            restore_metric_state(metric, saved["metric_state"], device)
            rows = saved["rows"]
            processed = int(saved["processed"])
            print(f"Resuming FID feature accumulation at {processed:,}/{total:,} images")
        else:
            print("Ignoring incompatible FID state from a different reference/method pair")
            state_path.unlink()
    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=args.wandb_run_id,
                resume="allow",
                mode=args.wandb_mode,
                name="thomas-kinkade-fid-30k",
                config={"milestones": milestones, "total_images": total},
            )
        except Exception as exc:
            print(f"W&B initialization failed; continuing locally: {exc}")

    for end in tqdm(range(processed + args.batch_size, total + args.batch_size, args.batch_size), desc="FID features"):
        batch_pairs = pairs[processed : min(end, total)]
        if not batch_pairs:
            break
        baseline = torch.stack([image_tensor(a) for a, _ in batch_pairs]).to(device)
        method = torch.stack([image_tensor(b) for _, b in batch_pairs]).to(device)
        metric.update(baseline, real=True)
        metric.update(method, real=False)
        processed += len(batch_pairs)
        reached = [m for m in milestones if m <= processed and not any(row["n_images"] == m for row in rows)]
        for milestone in reached:
            if milestone != processed:
                raise RuntimeError(
                    f"Batch size {args.batch_size} crossed milestone {milestone}. "
                    "Choose a batch size that divides every milestone."
                )
            fid = float(metric.compute().detach().cpu().item())
            previous = rows[-1]["fid"] if rows else None
            row = {
                "n_images": milestone,
                "fid": fid,
                "fid_change": None if previous is None else fid - previous,
            }
            rows.append(row)
            with open(output / "fid_progress.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
            pd.DataFrame(rows).to_csv(output / "fid_progress.csv", index=False)
            torch.save(
                {"metric_state": metric_state(metric), "rows": rows, "processed": processed, "signature": state_signature},
                state_path,
            )
            if wandb_run is not None:
                wandb_run.log(row)
            print(f"FID at {milestone:,} images: {fid:.6f}")
        if processed % 1000 == 0:
            torch.save(
                {"metric_state": metric_state(metric), "rows": rows, "processed": processed, "signature": state_signature},
                state_path,
            )

    if len(rows) != len(milestones):
        raise RuntimeError(f"Computed {len(rows)}/{len(milestones)} milestones")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([r["n_images"] for r in rows], [r["fid"] for r in rows], marker="o", linewidth=2)
    ax.axhline(14.50, color="#d95f02", linestyle="--", linewidth=1.5, label="Published SD v1.4: 14.50")
    ax.set_title("Thomas Kinkade SPACE: FID stabilization")
    ax.set_xlabel("Matched COCO images")
    ax.set_ylabel("FID vs. official COCO images (lower is better)")
    ax.grid(alpha=0.25)
    ax.legend()
    for row in rows:
        ax.annotate(f'{row["fid"]:.2f}', (row["n_images"], row["fid"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    plot_path = output / "fid_progress.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "metric": "FID",
        "reference": "official MS COCO 2014 validation images",
        "method": "SPACE Thomas Kinkade",
        "final_n_images": total,
        "final_fid": rows[-1]["fid"],
        "milestones": rows,
    }
    (output / "fid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run is not None:
        import wandb

        wandb_run.log({"fid_curve": wandb.Image(str(plot_path))})
        wandb_run.summary["final_fid_30k"] = rows[-1]["fid"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
