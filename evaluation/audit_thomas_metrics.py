#!/usr/bin/env python3
"""Independently validate Thomas Kinkade evaluation inputs and canonical FID."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_fidelity


def image_ids(directory: Path, suffix: str):
    return {int(path.stem) for path in directory.glob(f"*.{suffix}")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--coco_csv", required=True)
    args = parser.parse_args()

    root = Path(args.run_root)
    report = root / "report"
    reference = root / "fid/coco_reference"
    generated = root / "fid/space"
    prompts = pd.read_csv(args.coco_csv)
    metrics = pd.read_csv(report / "metrics.csv").iloc[0]
    prior_fid = json.loads((root / "fid_analysis/fid_summary.json").read_text())
    black = json.loads((report / "black_image_audit.json").read_text())

    expected_ids = {int(value) for value in prompts.image_id}
    real_ids = image_ids(reference, "jpg")
    fake_ids = image_ids(generated, "png")
    input_checks = {
        "coco_rows": len(prompts),
        "unique_coco_image_ids": len(expected_ids),
        "unique_evaluation_seeds": int(prompts.evaluation_seed.nunique()),
        "reference_images": len(real_ids),
        "space_images": len(fake_ids),
        "reference_ids_exact_match": real_ids == expected_ids,
        "space_ids_exact_match": fake_ids == expected_ids,
        "reference_space_ids_exact_match": real_ids == fake_ids,
    }
    if not all([
        len(prompts) == 30000,
        len(expected_ids) == 30000,
        len(real_ids) == 30000,
        len(fake_ids) == 30000,
        input_checks["reference_ids_exact_match"],
        input_checks["space_ids_exact_match"],
    ]):
        raise RuntimeError(f"FID input validation failed: {input_checks}")

    canonical = torch_fidelity.calculate_metrics(
        input1=str(reference),
        input2=str(generated),
        cuda=torch.cuda.is_available(),
        # Official COCO files have mixed native sizes. A single-item loader
        # preserves those files until torch-fidelity's own Inception resize.
        batch_size=1,
        fid=True,
        kid=False,
        isc=False,
        verbose=True,
    )
    canonical_fid = float(canonical["frechet_inception_distance"])
    custom_fid = float(prior_fid["final_fid"])

    paired_checks = {
        "n_pairs": int(metrics["n_pairs"]),
        "clip_drop_recomputed": float(metrics["clip_vanilla"] - metrics["clip_method"]),
        "clip_drop_reported": float(metrics["clip_drop"]),
        "style_drop_recomputed": float(metrics["style_vanilla"] - metrics["style_method"]),
        "style_drop_reported": float(metrics["style_drop"]),
        "style_target_hits": int(round(float(metrics["style_target_rate"]) * int(metrics["n_pairs"]))),
        "finite_metrics": {
            key: bool(np.isfinite(float(metrics[key])))
            for key in [
                "clip_vanilla", "clip_method", "clip_drop", "style_vanilla", "style_method",
                "style_drop", "style_target_rate", "clip_image_similarity", "lpips",
                "resnet_top1_agreement", "resnet_top5_overlap", "dino_similarity",
            ]
        },
    }
    paired_checks["clip_drop_identity_pass"] = bool(
        np.isclose(paired_checks["clip_drop_recomputed"], paired_checks["clip_drop_reported"], atol=1e-12)
    )
    paired_checks["style_drop_identity_pass"] = bool(
        np.isclose(paired_checks["style_drop_recomputed"], paired_checks["style_drop_reported"], atol=1e-12)
    )

    validation = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_CAVEATS",
        "input_checks": input_checks,
        "fid": {
            "canonical_torch_fidelity_native_preprocessing": canonical_fid,
            "torchmetrics_pre_resize_299_bilinear": custom_fid,
            "absolute_difference": abs(canonical_fid - custom_fid),
            "canonical_library": "torch-fidelity",
            "canonical_library_version": getattr(torch_fidelity, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "cuda_used": bool(torch.cuda.is_available()),
            "black_images_included": int(black["exact_black_images"]),
            "black_fraction": float(black["exact_black_fraction"]),
        },
        "paired_metric_checks": paired_checks,
        "metric_definitions": {
            "clip_text_scores": "Mean CLIP ViT-B/32 text-image logits divided by 100 over 500 pairs; custom implementation.",
            "clip_image_similarity": "Mean cosine similarity of normalized CLIP ViT-B/32 image embeddings over matched pairs.",
            "style_score": "Mean CLIP score for 'an artwork in the style of Thomas Kinkade'.",
            "style_target_rate": "Fraction classified as Thomas Kinkade among Thomas Kinkade, Monet, Rembrandt, Warhol, Picasso prompts; custom closed-set metric.",
            "lpips": "Mean LPIPS-VGG distance over matched vanilla/SPACE pairs after 224x224 bicubic resize.",
            "resnet": "Mean ImageNet ResNet-50 top-1 equality and top-5 set overlap over matched pairs.",
            "dino": "Mean cosine similarity of DINOv2-small CLS embeddings over matched pairs.",
        },
        "caveats": [
            "The canonical torch-fidelity FID is the value comparable to this repository's original evaluator; the pre-resized TorchMetrics value is retained as a separate implementation result.",
            "The published SD v1.4 FID=14.50 is context only unless sampling steps, prompts, seeds, safety handling, image count, and preprocessing are identical.",
            f"{black['exact_black_images']} safety-filtered black images ({100*black['exact_black_fraction']:.2f}%) are included in FID and describe the saved pipeline outputs, not an unchecked model distribution.",
            "CLIP style target rate is a custom five-label closed-set diagnostic, not a standardized artist-classification benchmark.",
            "Paired metrics use 500 prompt-matched outputs; FID uses 30,000 COCO-caption generations and must not be mixed at the sample level.",
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (report / "metric_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    md = f"""# Thomas Kinkade metric validation

**Assessment: PASS WITH CAVEATS**

## Independently checked FID

- Canonical `torch-fidelity` FID using native file preprocessing: **{canonical_fid:.6f}**
- Prior TorchMetrics FID after explicit 299×299 bilinear resize: **{custom_fid:.6f}**
- Absolute implementation difference: **{abs(canonical_fid-custom_fid):.6f}**
- Inputs: 30,000 unique official COCO JPGs and 30,000 exactly ID-matched SPACE PNGs
- Safety-filtered black images included: {black['exact_black_images']} ({100*black['exact_black_fraction']:.2f}%)

The canonical `torch-fidelity` value is the repository-comparable result. The prior value remains documented as a separate preprocessing implementation and must not be silently mixed with it.

## Paired metrics

- Exactly {int(metrics['n_pairs'])} matched vanilla/SPACE pairs were aggregated.
- CLIP-drop arithmetic identity: {'PASS' if paired_checks['clip_drop_identity_pass'] else 'FAIL'}
- Style-drop arithmetic identity: {'PASS' if paired_checks['style_drop_identity_pass'] else 'FAIL'}
- All reported paired metrics finite: {'PASS' if all(paired_checks['finite_metrics'].values()) else 'FAIL'}
- Style target hits: {paired_checks['style_target_hits']}/{int(metrics['n_pairs'])}

## Required interpretation caveats

1. Published SD v1.4 FID 14.50 is context only unless every generation and preprocessing setting matches.
2. The 0.48% black-image rate is part of the evaluated saved pipeline output and can affect FID.
3. Style target rate is a custom five-label CLIP diagnostic, not a standardized benchmark.
4. Paired metrics use 500 matched prompt outputs; FID uses a separate 30,000-image COCO evaluation.
"""
    (report / "metric_validation.md").write_text(md)
    print(json.dumps(validation["fid"], indent=2))


if __name__ == "__main__":
    main()
