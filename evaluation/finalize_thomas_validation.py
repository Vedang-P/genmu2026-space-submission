#!/usr/bin/env python3
"""Promote validated canonical FID while preserving the prior implementation result."""

import json
from pathlib import Path

import pandas as pd


root = Path("results/thomas_kinkade_30k")
report = root / "report"
validation = json.loads((report / "metric_validation.json").read_text())
canonical = float(validation["fid"]["canonical_torch_fidelity_native_preprocessing"])
prior = float(validation["fid"]["torchmetrics_pre_resize_299_bilinear"])

metrics_path = report / "metrics.csv"
metrics = pd.read_csv(metrics_path)
metrics["fid"] = canonical
metrics["fid_implementation"] = "torch-fidelity-0.3.0-native-input-batch1"
metrics["fid_pre_resized_torchmetrics"] = prior
metrics.to_csv(metrics_path, index=False)

manifest_path = report / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["final_fid"] = canonical
manifest["fid_implementation"] = "torch-fidelity 0.3.0, native input, batch_size=1"
manifest["prior_pre_resized_torchmetrics_fid"] = prior
for name in ["metric_validation.json", "metric_validation.md"]:
    if name not in manifest["report_files"]:
        manifest["report_files"].append(name)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

summary_path = report / "summary.md"
summary = summary_path.read_text()
summary = summary.replace("- FID (30k): 14.089452", f"- Canonical torch-fidelity FID (30k): {canonical:.6f}\n- Prior pre-resized TorchMetrics FID (noncanonical): {prior:.6f}")
summary = summary.replace("- Published vanilla SD v1.4 FID: 14.50", "- Published vanilla SD v1.4 FID: 14.50 (context only; generation settings are not proven identical)")
if "metric_validation.md" not in summary:
    summary += "\n- `metric_validation.md`: independent implementation and input audit\n"
summary_path.write_text(summary)

print(f"Promoted canonical FID {canonical:.6f}; retained prior result {prior:.6f}")
