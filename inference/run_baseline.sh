#!/bin/bash
# ============================================================
# run_baseline.sh — Generate baseline images with stock SD v1.4
# Run all 3 prompt CSVs in parallel (one process each).
# On A40 (48GB), 3x SD v1.4 fp16 ≈ 12GB — well within limits.
# ============================================================

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/baseline logs

echo "============================================================"
echo "  ESD Replication: Baseline Image Generation (SD v1.4)"
echo "  Launching 3 parallel jobs..."
echo "============================================================"

python inference/generate_esd_old.py \
  --prompts_path data/kelly_prompts.csv \
  --save_path results/baseline/kelly \
  2>&1 | tee logs/baseline_kelly.log &
PID1=$!
echo "  [1/3] Kelly McKernan  — PID $PID1  (logs/baseline_kelly.log)"

python inference/generate_esd_old.py \
  --prompts_path data/vangogh_prompts.csv \
  --save_path results/baseline/vangogh \
  2>&1 | tee logs/baseline_vangogh.log &
PID2=$!
echo "  [2/3] Van Gogh        — PID $PID2  (logs/baseline_vangogh.log)"

python inference/generate_esd_old.py \
  --prompts_path data/short_niche_art_prompts.csv \
  --save_path results/baseline/niche \
  2>&1 | tee logs/baseline_niche.log &
PID3=$!
echo "  [3/3] Niche artists   — PID $PID3  (logs/baseline_niche.log)"

echo ""
echo "  All 3 baseline jobs running. Waiting for completion..."
echo "  Monitor: tail -f logs/baseline_kelly.log"
echo ""

wait $PID1; echo "  [1/3] Kelly baseline DONE"
wait $PID2; echo "  [2/3] VanGogh baseline DONE"
wait $PID3; echo "  [3/3] Niche baseline DONE"

echo ""
echo "============================================================"
echo "  ALL BASELINE JOBS COMPLETE"
echo "  Images saved to: results/baseline/"
echo "============================================================"
