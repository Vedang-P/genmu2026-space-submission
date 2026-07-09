#!/bin/bash
# Resumable Thomas Kinkade SPACE experiment: 50x10 prompt pairs and COCO-30k FID.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/baselines/use_venv.sh"
source "$ROOT/baselines/progress.sh"
cd "$ROOT"

DEVICE="${DEVICE:-cuda:0}"
RUN_ROOT="${RUN_ROOT:-results/thomas_kinkade_30k}"
PROMPTS="data/thomas_kinkade_prompts.csv"
COCO_PROMPTS="data/coco_30k.csv"
CHECKPOINT="space-models/sd/space-Thomas_Kinkade.safetensors"
REFERENCE_DIR="$RUN_ROOT/fid/coco_reference"
WANDB_PROJECT="${WANDB_PROJECT:-space-thomas-kinkade}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
FID_IMAGES="${FID_IMAGES:-30000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-8}"
FID_STEPS="${FID_STEPS:-20}"
PROMPT_STEPS="${PROMPT_STEPS:-20}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
EXPECTED_PROMPT_IMAGES=$((50 * NUM_SAMPLES))
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
COMMIT_RESULTS="${COMMIT_RESULTS:-0}"

mkdir -p "$RUN_ROOT"/{progress,logs,prompt_pairs,fid,fid_analysis,report} space-models/sd
export WANDB_DIR="${WANDB_DIR:-$ROOT/$RUN_ROOT/wandb}"
mkdir -p "$WANDB_DIR"

wandb_args=(--wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE")
if [ -n "$WANDB_ENTITY" ]; then
  wandb_args+=(--wandb_entity "$WANDB_ENTITY")
fi

count_pngs() {
  find "$1" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' '
}

count_reference_images() {
  find "$1" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | wc -l | tr -d ' '
}

require_count() {
  local label="$1" dir="$2" expected="$3" actual
  actual="$(count_pngs "$dir")"
  if [ "$actual" != "$expected" ]; then
    echo "$label has $actual PNGs; expected $expected in $dir" >&2
    exit 1
  fi
}

progress_banner "Thomas Kinkade SPACE 30k experiment"
echo "Run root: $ROOT/$RUN_ROOT"
echo "W&B project: $WANDB_PROJECT ($WANDB_MODE)"
echo "FID batch size: $FID_BATCH_SIZE"

# Download/extract the official real-image COCO reference concurrently with GPU work.
"$PYTHON_BIN" evaluation/download_coco_reference.py \
  --prompts_path "$COCO_PROMPTS" \
  --output_dir "$REFERENCE_DIR" \
  --cache_dir "$RUN_ROOT/download_cache" \
  --progress_path "$RUN_ROOT/progress/coco_reference.jsonl" \
  > "$RUN_ROOT/logs/coco_reference.log" 2>&1 &
COCO_REFERENCE_PID=$!
echo "Official COCO reference download PID: $COCO_REFERENCE_PID"

if [ "$SKIP_PREFLIGHT" != "1" ]; then
  bash baselines/preflight_env.sh
fi

"$PYTHON_BIN" - <<'PY'
import pandas as pd
target = pd.read_csv("data/thomas_kinkade_prompts.csv")
coco = pd.read_csv("data/coco_30k.csv")
assert len(target) == 50, f"expected 50 target prompts, got {len(target)}"
assert target.case_number.tolist() == list(range(50)), "target case numbers must be 0..49"
assert target.prompt.nunique() == 50, "target prompts must be unique"
assert len(coco) == 30000, f"expected 30,000 COCO prompts, got {len(coco)}"
print("Prompt validation passed: 50 target prompts and 30,000 COCO prompts")
PY

if [ ! -f "$CHECKPOINT" ]; then
  progress_step 1 7 "train SPACE for Thomas Kinkade"
  "$PYTHON_BIN" training/space_sd.py \
    --erase_concept "Thomas Kinkade" \
    --target_prompts_path "$PROMPTS" \
    --target_artist_filter "Thomas Kinkade" \
    --preserve_prompts_path "$COCO_PROMPTS" \
    --neutral_art_template "a high quality painting of {content}" \
    --neutral_image_template "a high quality image of {content}" \
    --style_basis_rank "${STYLE_BASIS_RANK:-4}" \
    --saliency_topk_blocks "${SALIENCY_TOPK_BLOCKS:-0.25}" \
    --trajectory_bands "${TRAJECTORY_BANDS:-0.2,0.5,0.8}" \
    --robust_prompt_mode "${ROBUST_PROMPT_MODE:-full}" \
    --vulnerable_preserve_k "${VULNERABLE_PRESERVE_K:-3}" \
    --lora_rank "${LORA_RANK:-8}" \
    --stage1_steps "${SPACE_STAGE1_STEPS:-300}" \
    --stage2_steps "${SPACE_STAGE2_STEPS:-100}" \
    --lr "${SPACE_LR:-1e-4}" \
    --guidance_scale "${SPACE_TRAIN_GUIDANCE_SCALE:-3.0}" \
    --num_inference_steps "${SPACE_TRAIN_STEPS:-50}" \
    --alpha_art "${SPACE_ALPHA_ART:-0.30}" \
    --erase_scale "${SPACE_ERASE_SCALE:-1.0}" \
    --lambda_style "${SPACE_LAMBDA_STYLE:-0.8}" \
    --lambda_content "${SPACE_LAMBDA_CONTENT:-1.0}" \
    --lambda_image "${SPACE_LAMBDA_IMAGE:-0.5}" \
    --lambda_preserve "${SPACE_LAMBDA_PRESERVE:-0.20}" \
    --lambda_vulnerable "${SPACE_LAMBDA_VULNERABLE:-0.30}" \
    --lambda_lora "${SPACE_LAMBDA_LORA:-1e-4}" \
    --preserve_limit "${PRESERVE_LIMIT:-256}" \
    --save_path space-models/sd \
    --exp_name Thomas_Kinkade \
    --device "$DEVICE" \
    --progress_dir "$RUN_ROOT/progress/training" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --wandb_run_id thomas-kinkade-space-training \
    "${wandb_args[@]}" 2>&1 | tee "$RUN_ROOT/logs/training.log"
else
  echo "Reusing checkpoint: $CHECKPOINT"
fi

progress_step 2 7 "generate 500 vanilla prompt images"
"$PYTHON_BIN" inference/generate_images.py \
  --base_model CompVis/stable-diffusion-v1-4 \
  --prompts_path "$PROMPTS" \
  --save_path "$RUN_ROOT/prompt_pairs" \
  --model_name_override baseline \
  --num_samples "$NUM_SAMPLES" \
  --num_inference_steps "$PROMPT_STEPS" \
  --guidance_scale 7.5 \
  --device "$DEVICE" \
  --progress_path "$RUN_ROOT/progress/prompt_baseline.jsonl" \
  --wandb_run_id thomas-kinkade-prompt-baseline \
  "${wandb_args[@]}" 2>&1 | tee "$RUN_ROOT/logs/prompt_baseline.log"

progress_step 3 7 "generate 500 SPACE prompt images"
"$PYTHON_BIN" inference/generate_images.py \
  --base_model CompVis/stable-diffusion-v1-4 \
  --esd_path "$CHECKPOINT" \
  --prompts_path "$PROMPTS" \
  --save_path "$RUN_ROOT/prompt_pairs" \
  --model_name_override space \
  --num_samples "$NUM_SAMPLES" \
  --num_inference_steps "$PROMPT_STEPS" \
  --guidance_scale 7.5 \
  --device "$DEVICE" \
  --progress_path "$RUN_ROOT/progress/prompt_space.jsonl" \
  --wandb_run_id thomas-kinkade-prompt-space \
  "${wandb_args[@]}" 2>&1 | tee "$RUN_ROOT/logs/prompt_space.log"
require_count "Vanilla prompt set" "$RUN_ROOT/prompt_pairs/baseline" "$EXPECTED_PROMPT_IMAGES"
require_count "SPACE prompt set" "$RUN_ROOT/prompt_pairs/space" "$EXPECTED_PROMPT_IMAGES"

progress_step 4 7 "generate $FID_IMAGES SPACE COCO images"
"$PYTHON_BIN" evaluation/generate_fid_samples.py \
  --base_model CompVis/stable-diffusion-v1-4 \
  --esd_path "$CHECKPOINT" \
  --prompts_path "$COCO_PROMPTS" \
  --save_path "$RUN_ROOT/fid/space" \
  --n_images "$FID_IMAGES" \
  --batch_size "$FID_BATCH_SIZE" \
  --num_inference_steps "$FID_STEPS" \
  --guidance_scale 7.5 \
  --device "$DEVICE" \
  --progress_path "$RUN_ROOT/progress/fid_space.jsonl" \
  --wandb_run_id thomas-kinkade-fid-space \
  "${wandb_args[@]}" 2>&1 | tee "$RUN_ROOT/logs/fid_space.log"
require_count "SPACE FID set" "$RUN_ROOT/fid/space" "$FID_IMAGES"

if [ "$FID_IMAGES" != "30000" ]; then
  echo "Full report requires FID_IMAGES=30000; generation smoke run completed at $FID_IMAGES." >&2
  exit 0
fi

progress_step 5 7 "validate official COCO reference"
wait "$COCO_REFERENCE_PID"
REFERENCE_COUNT="$(count_reference_images "$REFERENCE_DIR")"
if [ "$REFERENCE_COUNT" != "30000" ]; then
  echo "Official COCO reference has $REFERENCE_COUNT images; expected 30000" >&2
  exit 1
fi

progress_step 6 7 "compute cumulative FID milestones and paired metrics"
"$PYTHON_BIN" evaluation/analyze_fid_progress.py \
  --prompts_path "$COCO_PROMPTS" \
  --baseline_dir "$REFERENCE_DIR" \
  --method_dir "$RUN_ROOT/fid/space" \
  --output_dir "$RUN_ROOT/fid_analysis" \
  --device "$DEVICE" \
  --wandb_run_id thomas-kinkade-fid-analysis \
  "${wandb_args[@]}" 2>&1 | tee "$RUN_ROOT/logs/fid_analysis.log"

rm -f results/evaluation/metrics_cache_v5_thomas_kinkade.json
"$PYTHON_BIN" evaluation/evaluate.py \
  --only-artist "Thomas Kinkade" \
  --method-keys space \
  --expected-pairs 500 \
  --skip-fid 2>&1 | tee "$RUN_ROOT/logs/paired_evaluation.log"
"$PYTHON_BIN" evaluation/make_comparison_grids.py --only-artist "Thomas Kinkade" --methods SPACE

progress_step 7 7 "build compact report bundle"
cp results/provenance/space/train_space-Thomas_Kinkade.json "$RUN_ROOT/report/training_provenance.json"
cp "$RUN_ROOT/progress/training/run_config.json" "$RUN_ROOT/report/run_config.json"
"$PYTHON_BIN" evaluation/finalize_thomas_kinkade_report.py \
  --run_root "$RUN_ROOT" \
  --paired_metrics results/evaluation/metrics.csv \
  --comparison_grid results/comparisons/thomas_kinkade.png

if [ "$COMMIT_RESULTS" = "1" ]; then
  git add "$RUN_ROOT/report"
  if git diff --cached --quiet; then
    echo "No report changes to commit."
  else
    git commit -m "Add Thomas Kinkade SPACE 30k results"
  fi
fi

echo "Experiment complete. Report: $RUN_ROOT/report/summary.md"
