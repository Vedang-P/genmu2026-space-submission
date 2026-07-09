# Thomas Kinkade SPACE 30k RunPod Runbook

## Recommended Pod

- RunPod **Secure Cloud on-demand Pod**
- **1x A100 SXM 80 GB** (A100 PCIe 80 GB is also suitable)
- Official RunPod PyTorch image with SSH enabled
- **30 GB container disk**
- **120 GB network volume**, mounted at `/workspace`
- At least 16 vCPU and 100 GB system RAM when available

The network volume is important: it survives Pod deletion and stores the repository, Python environment, checkpoints, W&B state, 1,000 prompt images, 30,000 SPACE COCO generations, and the official COCO reference. Do not use Spot for the first full run.

## Credentials Needed

Export these inside the Pod; do not commit them:

```bash
export HF_TOKEN='...'
export WANDB_API_KEY='...'
```

The Hugging Face account must have accepted the `CompVis/stable-diffusion-v1-4` model terms. Git push access is needed only if the Pod should push the final result commit.

## Bootstrap

Clone under the persistent mount, then install the repository's pinned environment:

```bash
cd /workspace
git clone https://github.com/Vedang-P/space-claude-implementation.git
cd space-claude-implementation/erasing
bash baselines/bootstrap_runpod_env.sh
source .venv/bin/activate
huggingface-cli login --token "$HF_TOKEN"
wandb login "$WANDB_API_KEY"
```

## Preflight

```bash
cd /workspace/space-claude-implementation/erasing
bash baselines/preflight_env.sh
nvidia-smi
```

For a cheap end-to-end generation check before the full run:

```bash
FID_IMAGES=16 NUM_SAMPLES=1 SPACE_STAGE1_STEPS=1 SPACE_STAGE2_STEPS=0 \
FID_BATCH_SIZE=4 WANDB_MODE=offline \
bash run_thomas_kinkade_30k.sh
```

After this check, remove the smoke checkpoint and outputs before the full run because the harness deliberately reuses existing files:

```bash
rm -f space-models/sd/space-Thomas_Kinkade.safetensors
rm -rf results/thomas_kinkade_30k
```

## Full Run

Run inside `tmux` so SSH disconnects do not stop the job:

```bash
cd /workspace/space-claude-implementation/erasing
tmux new -s thomas-kinkade
source .venv/bin/activate
WANDB_PROJECT=space-thomas-kinkade \
FID_BATCH_SIZE=8 \
COMMIT_RESULTS=1 \
bash run_thomas_kinkade_30k.sh
```

If batch 8 runs out of memory, restart with `FID_BATCH_SIZE=4`; all completed PNGs are skipped. If the process stops, rerun the same command. Prompt and FID generation resume file-by-file, the official COCO download/extraction resumes, W&B uses stable resumable run IDs, training metrics are fsynced each step, and training snapshots are written every 50 steps.

## Outputs

Bulk outputs remain under:

```text
results/thomas_kinkade_30k/
```

The commit-safe bundle is:

```text
results/thomas_kinkade_30k/report/
```

With `COMMIT_RESULTS=1`, the harness commits only that compact report directory. It never pushes automatically. Push after inspection:

```bash
git status
git push origin main
```
