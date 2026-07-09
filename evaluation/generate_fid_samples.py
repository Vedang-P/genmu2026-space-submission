#!/usr/bin/env python3
"""Generate N diverse images from coco_30k.csv for FID computation.

Saves one image per prompt to --save_path/{image_id}.png.
Skips images that already exist so re-runs are safe.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
from diffusers import DiffusionPipeline

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.esd_checkpoint import apply_esd_checkpoint


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def generate_fid_samples(
    base_model,
    esd_path,
    prompts_path,
    save_path,
    n_images=2000,
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    guidance_scale=7.5,
    num_inference_steps=20,
    component=None,
    progress_path=None,
    progress_interval=100,
    wandb_project=None,
    wandb_run_id=None,
    wandb_entity=None,
    wandb_mode="disabled",
    batch_size=1,
):
    os.makedirs(save_path, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    if esd_path is not None:
        if esd_path.endswith(".pt"):
            unet_weights = torch.load(esd_path, map_location="cpu", weights_only=False)
            pipe.unet.load_state_dict(unet_weights)
            pipe.unet.eval()
            print(f"Loaded .pt checkpoint into pipe.unet")
        else:
            metadata, resolved_component, _ = apply_esd_checkpoint(
                pipe, esd_path, device="cpu", component_name=component
            )
            if metadata.get("base_model_id") and metadata["base_model_id"] != base_model:
                print(f"Warning: checkpoint was trained on {metadata['base_model_id']}, running on {base_model}")
            print(f"Loaded checkpoint into pipe.{resolved_component}")

    df = pd.read_csv(prompts_path)
    df = df.head(n_images)

    already = sum(1 for _, row in df.iterrows() if os.path.exists(os.path.join(save_path, f"{row.image_id}.png")))
    if already:
        print(f"Resuming: {already}/{len(df)} already done, skipping those.")

    started = time.time()
    completed = already
    wandb_run = None
    if wandb_project and wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                id=wandb_run_id,
                resume="allow",
                mode=wandb_mode,
                name=f"fid-images-{os.path.basename(save_path)}",
                config={"n_images": len(df), "checkpoint": esd_path or "vanilla"},
            )
        except Exception as exc:
            print(f"W&B initialization failed; continuing locally: {exc}")

    def log_progress(force=False):
        if not force and completed % max(progress_interval, 1) != 0:
            return
        elapsed = time.time() - started
        payload = {
            "phase": "fid_generation",
            "completed_images": completed,
            "total_images": len(df),
            "elapsed_seconds": elapsed,
            "images_per_second": (completed - already) / elapsed if elapsed > 0 else 0.0,
        }
        if progress_path:
            os.makedirs(os.path.dirname(os.path.abspath(progress_path)), exist_ok=True)
            with open(progress_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
                f.flush()
                os.fsync(f.fileno())
        if wandb_run is not None:
            wandb_run.log(payload)

    log_progress(force=True)
    pending = [row for _, row in df.iterrows() if not os.path.exists(os.path.join(save_path, f"{row.image_id}.png"))]
    from tqdm import tqdm
    for start in tqdm(range(0, len(pending), batch_size), desc=os.path.basename(save_path)):
        rows = pending[start : start + batch_size]
        images = pipe(
            [str(row.prompt) for row in rows],
            generator=[make_generator(int(row.evaluation_seed)) for row in rows],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
        for row, image in zip(rows, images):
            out_path = os.path.join(save_path, f"{row.image_id}.png")
            image.save(out_path)
            completed += 1
            log_progress()

    print(f"Done. {len(df)} images in {save_path}")
    log_progress(force=True)
    if wandb_run is not None:
        wandb_run.summary["completed_images"] = completed
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FID reference images from COCO prompts.")
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--esd_path", default=None, help="Optional checkpoint to apply")
    parser.add_argument("--component", default=None)
    parser.add_argument("--prompts_path", required=True, help="Path to coco_30k.csv")
    parser.add_argument("--save_path", required=True, help="Output directory")
    parser.add_argument("--n_images", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--progress_path", default=None)
    parser.add_argument("--progress_interval", type=int, default=100)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_id", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--batch_size", type=int, default=1, help="generation batch size; reduce if CUDA OOM occurs")
    args = parser.parse_args()

    generate_fid_samples(
        base_model=args.base_model,
        esd_path=args.esd_path,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        n_images=args.n_images,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        component=args.component,
        progress_path=args.progress_path,
        progress_interval=args.progress_interval,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        batch_size=args.batch_size,
    )
