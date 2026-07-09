# SPACE: Style-Preserving Attribute-Constrained Erasure

## What SPACE Is

SPACE is a preservation-first method for artist/style unlearning in Stable Diffusion v1.4. The acronym stands for **Style-Preserving Attribute-Constrained Erasure**. At a high level, SPACE edits a text-to-image diffusion model so that prompts containing a target artist cue no longer reproduce that recognizable style, while the semantic content of the prompt, the model's behavior on unrelated prompts, and nearby non-target artistic styles remain as intact as possible. The intended output is not artist replacement and not coarse degradation toward a generic low-quality image distribution; it is **neutralized style suppression under explicit preservation constraints**. In the current codebase, the primary implemented method is **SPACE**, and this document describes that version only, using the actual implementation in [training/space_sd.py](../training/space_sd.py) and [core/space_trainer.py](../core/space_trainer.py) as the ground truth.

## Problem Setting

Let `p_t` be a prompt that contains a target artist or style attribute, such as:

```text
Bedroom in Arles by Vincent van Gogh
```

The desired edited-model output should still depict a coherent bedroom scene. It should not collapse semantically, and it should not intentionally convert the prompt into another named artist's style. This makes the task different from style transfer, concept replacement, or broad deletion. The paper-facing objective of SPACE is:

```text
given a target style-bearing prompt p_t,
produce outputs that preserve content and broad image quality
while suppressing the target style signal.
```

Formally, SPACE is designed to optimize an erasure-preservation tradeoff. If `f_base` denotes the base Stable Diffusion model and `f_space` denotes the edited model, then for a target style concept `c*`, the ideal behavior is:

```text
target-style score(f_space(p_t)) << target-style score(f_base(p_t))
content score(f_space(p_t))      ~= content score(f_base(p_t))
quality score(f_space(p_t))      ~= quality score(f_base(p_t))
nearby-style damage              small
general-prompt damage            small
```

This is the central design lens of SPACE: **stronger suppression must not come from indiscriminate destruction of content or general utility**.

## Method Overview

SPACE combines four core design choices. First, it factorizes each target prompt into several prompt-conditioned views that separate content, target style, neutral art structure, and generic image structure. Second, it constructs a teacher target that explicitly subtracts the target style residual while anchoring the prediction toward a neutral content-preserving point. Third, it localizes the update by inserting trainable LoRA adapters only into saliency-selected cross-attention projections of the U-Net. Fourth, it trains this edit across several denoising trajectory bands while replaying preservation prompts and vulnerable nearby styles. The result is a structured teacher-student procedure in which the student is the base U-Net plus localized low-rank adapters, and the teacher is the same U-Net evaluated with those adapters disabled.

## Methodology

### Notation

Let:

```text
x_t          = noisy latent at denoising timestep t
eps_theta    = noise-prediction U-Net of the base model
eps_tilde    = noise-prediction U-Net with SPACE LoRA adapters enabled
tau(p)       = text embedding of prompt p
T            = set of trajectory-band timestep indices
```

We use four prompt-conditioned views per target prompt:

```text
p_t  = original style-bearing target prompt
p_c  = content-only prompt with target style phrase removed
p_a  = neutral art anchor prompt, "a high quality painting of {content}"
p_g  = generic image anchor prompt, "a high quality image of {content}"
```

These are constructed in the trainer by `strip_concept_from_prompt(...)`, `build_anchor_prompt(...)`, and `build_prompt_variants(...)` in [core/space_trainer.py](../core/space_trainer.py).

### Prompt Family Construction

SPACE does not assume that the target style can be isolated by one literal string match. The trainer builds a prompt family around the target prompt by expanding robust variants such as:

```text
by {artist}
in the style of {artist}
inspired by {artist}
as painted by {artist}
```

and shortened or transformed artist references where relevant. This family-level treatment matters because it prevents the method from overfitting to one prompt surface form while leaving paraphrased style references untouched. In implementation terms, the target supervision is therefore defined over a set of prompt-conditioned style views, not a single prompt tokenization.

### Teacher Signal Construction

The teacher target is implemented in `make_space_teacher(...)` in [core/space_trainer.py](../core/space_trainer.py). At a fixed latent `x_t` and timestep `t`, the base U-Net is queried under the target, content, art-anchor, and image-anchor prompts:

```text
base_target  = eps_theta(x_t, tau(p_t))
base_content = eps_theta(x_t, tau(p_c))
base_art     = eps_theta(x_t, tau(p_a))
base_image   = eps_theta(x_t, tau(p_g))
```

The crucial construction is:

```text
neutral_anchor = (1 - alpha_art) * base_content
               + alpha_art       * base_art

style_direction = base_target - base_content

teacher = neutral_anchor - erase_scale * style_direction
```

With the default settings from [training/space_sd.py](../training/space_sd.py),

```text
alpha_art   = 0.30
erase_scale = 1.0
```

so the neutral anchor is:

```text
neutral_anchor = 0.7 * base_content + 0.3 * base_art
```

This is the heart of SPACE. The term `style_direction = base_target - base_content` estimates the target-style residual under the base model. Rather than merely distilling toward a generic anchor, SPACE explicitly subtracts that style residual. The teacher therefore combines two impulses:

1. **anchor toward content-preserving neutral art structure**
2. **move away from the target style direction**

This is stronger and more structured than simply forcing the target prompt to imitate `base_content` or `base_art` alone.

A critical implementation note is that `base_image` is **not** part of the teacher construction. It is computed separately and used later in the image-preservation loss. Any description that routes `base_image` into the teacher target is inaccurate for SPACE.

### Teacher-Student Structure

SPACE is best viewed as a **LoRA-off / LoRA-on** training scheme over a shared backbone. The trainer does not instantiate a separate frozen teacher network. Instead, the same U-Net is evaluated in two modes:

```text
teacher/base path: set_adapters_enabled(adapters, False)
student path:      set_adapters_enabled(adapters, True)
```

Thus, the “teacher” is the base model's prediction path with no adapter contribution, while the student is the adapted model. This distinction matters because the method is not distilling from an external frozen expert; it is editing the base model relative to its own unedited predictions.

### Localized LoRA Editing

SPACE does not update the full U-Net. It injects trainable low-rank adapters only into a subset of cross-attention linear projections selected by saliency. Candidate modules are gathered by `select_cross_attention_linears(...)`, ranked by `ranked_saliency_modules(...)`, and truncated by `resolve_topk_modules(...)`. The adapters are then inserted by `inject_lora_adapters(...)`.

If `W in R^(d_out x d_in)` is a frozen linear map, the LoRA parameterization at that layer is:

```text
A in R^(r x d_in)
B in R^(d_out x r)
delta(x) = B(Ax)
h = W x + (alpha / r) * delta(x)
```

In the actual implementation:

```text
r      = lora_rank = 8
alpha  = r
alpha / r = 1.0
```

`A` (`lora_down`) is initialized with Kaiming uniform initialization and `B` (`lora_up`) is initialized to zero. Only `A` and `B` are optimized. The LoRA branch is a **parallel residual path on input `x`**, not a post-processing block on the output of `W`.

### Saliency-Based Module Selection

The edited module set is data-dependent rather than hardcoded. The trainer probes a small set of target families and trajectory bands, computes a teacher-matching loss, and backpropagates it into the candidate cross-attention linear weights of the base U-Net. The accumulated gradient norm defines a saliency score:

```text
s(m) = sum over probe families and probe bands of || grad_{W_m} L_probe ||_2
```

The top-ranked modules are selected according to `saliency_topk_blocks`, which defaults to:

```text
saliency_topk_blocks = 0.25
```

This means SPACE should not be described as “LoRA on 16 layers” or any other fixed layer count unless that count is measured from one exact run. In the actual code, the edited set is saliency-selected and may vary with configuration.

### Timestep-Band Supervision

SPACE constrains the edit across multiple denoising regions instead of at one sampled timestep. The trajectory bands are parsed by `trajectory_indices(...)` from:

```text
trajectory_bands = "0.2,0.5,0.8"
```

which correspond to a set of timestep indices `T`. At each training step, the objective accumulates per-band losses and averages them:

```text
L_bands = (1 / |T|) * sum_{t in T} L_t
```

This band-wise averaging is important because a diffusion edit that only works at one point in the trajectory may still allow the target style to leak elsewhere in the reverse process. SPACE instead regularizes the edit across early, middle, and late denoising regions.

### Loss Design

The total loss at a training step is:

```text
L_total =
  (1 / |T|) * sum_{t in T} [
      L_erase(t)
    + lambda_style      * L_style(t)
    + lambda_content    * L_content(t)
    + lambda_image      * L_image(t)
    + lambda_preserve   * L_preserve(t)
    + lambda_vulnerable * L_vulnerable(t)
  ]
  + lambda_lora * L_reg
```

with current defaults:

```text
lambda_style      = 0.8
lambda_content    = 1.0
lambda_image      = 0.5
lambda_preserve   = 0.20
lambda_vulnerable = 0.30
lambda_lora       = 1e-4
```

Each term serves a distinct purpose.

#### Erase Loss

```text
L_erase(t) = || eps_tilde(x_t, tau(p_t)) - teacher ||_2^2
```

This is the main erasure term. It moves the student target prediction toward the SPACE teacher.

#### Content Preservation

```text
L_content(t) = || eps_tilde(x_t, tau(p_c)) - eps_theta(x_t, tau(p_c)) ||_2^2
```

This stabilizes content-only behavior and prevents the edited model from drifting too far from the base model on the semantic core of the prompt.

#### Generic Image Preservation

```text
L_image(t) = || eps_tilde(x_t, tau(p_g)) - eps_theta(x_t, tau(p_g)) ||_2^2
```

This term preserves broad image utility and discourages over-specialization toward “painting-like” structure alone.

#### Style Suppression as Subspace Control

The style term is easy to mischaracterize. It is **not** just another direct MSE to the teacher. Instead, the trainer builds a low-rank style basis from several target-family variants:

```text
Delta_i = eps_theta(x_t, tau(p_t^(i))) - eps_theta(x_t, tau(p_c^(i)))
```

for several prompt variants `i`. These deltas are stacked and centered, and the trainer computes a rank-`k` basis `U_k` by singular value decomposition in `build_style_basis(...)`. The student residual is then penalized by its projected energy:

```text
r_student = eps_tilde(x_t, tau(p_t)) - eps_theta(x_t, tau(p_c))
L_style(t) = || U_k U_k^T r_student ||_2^2
```

In implementation, this is handled by `build_style_basis(...)` and `projection_energy(...)`. The consequence is that SPACE suppresses a **style subspace**, not just one residual vector.

#### General Preserve Replay

SPACE also preserves behavior on non-target prompts drawn from a generic prompt pool, by default `data/coco_30k.csv`. Let `p_pres` be one such preserve prompt. The trainer samples a separate preserve latent:

```text
x_t^pres = sample_xt(..., seed + 1, ...)
```

and computes:

```text
L_preserve(t) =
  || eps_tilde(x_t^pres, tau(p_pres)) - eps_theta(x_t^pres, tau(p_pres)) ||_2^2
```

This means `L_preserve` is not computed on the same latent as the target branch.

#### Vulnerable-Style Preserve Replay

SPACE explicitly mines nearby styles from an internal artist bank using CLIP text similarity. If `p_vuln` denotes a prompt rewritten toward a vulnerable nearby style, the trainer samples another separate preserve latent:

```text
x_t^vuln = sample_xt(..., seed + 2, ...)
```

and computes:

```text
L_vulnerable(t) =
  || eps_tilde(x_t^vuln, tau(p_vuln)) - eps_theta(x_t^vuln, tau(p_vuln)) ||_2^2
```

This is one of the major preservation mechanisms that distinguishes SPACE from simpler erase-only methods.

#### LoRA Regularization

The adapter regularization term penalizes the mean squared magnitude of LoRA parameters:

```text
L_reg = mean( ||A_m||_F^2, ||B_m||_F^2 over edited modules m )
```

as implemented by `adapter_regularization(...)`. This keeps the edit localized and discourages uncontrolled adapter growth.

### Preserve Replay Uses Separate Latents

A critical structural fact in the code is that **not all losses are attached to one shared latent `x_t`**. The target/content/image/style terms are computed around one sampled target latent, but the general-preserve and vulnerable-preserve terms each use separately sampled replay latents with offset seeds. Any diagram or description that compresses all six losses into one shared latent branch is therefore methodologically inaccurate for SPACE.

### Two-Stage Curriculum

The trainer uses an explicit two-stage schedule:

```text
stage1_steps = 300
stage2_steps = 100
```

and computes:

```text
stage2 = step >= stage1_steps
```

During the earlier stage, supervision is narrower. In the later stage, the model sees a broader family of prompt variants for the target concept and corresponding style basis construction. Intuitively, stage 1 stabilizes the core edit, while stage 2 broadens robustness to paraphrase and indirect style mention.

## Implementation Mapping to the Codebase

The public configuration surface for SPACE lives in [training/space_sd.py](../training/space_sd.py), which defines the `SPACEConfig` arguments and forwards them into the trainer. The core implementation lives in [core/space_trainer.py](../core/space_trainer.py). The major implementation blocks are:

- prompt decomposition and robust prompt families:
  - `strip_concept_from_prompt(...)`
  - `build_prompt_variants(...)`
  - `build_anchor_prompt(...)`
  - `load_target_families(...)`
- teacher construction:
  - `make_space_teacher(...)`
- saliency ranking and localization:
  - `select_cross_attention_linears(...)`
  - `ranked_saliency_modules(...)`
  - `inject_lora_adapters(...)`
- style subspace computation:
  - `build_style_basis(...)`
  - `projection_energy(...)`
- preserve replay:
  - `load_preserve_prompts(...)`
  - `encode_preserve_prompts(...)`
  - `mine_vulnerable_artists(...)`
- main training loop:
  - `run_space_training(...)`

The trainer also records checkpoint metadata and provenance through:

- `build_space_metadata(...)`
- `record_space_provenance(...)`
- partial and final checkpoint helpers

which makes the method reproducible as a research artifact rather than just an ad hoc experiment.

## Diagram Accuracy Notes

Several compact diagrams can inadvertently misstate SPACE if they flatten away implementation detail. For this codebase, the correct reading is:

- there is **no separate frozen teacher network**; the teacher is the shared U-Net with LoRA disabled
- `base_image` is **not** part of teacher construction
- the preserve losses do **not** reuse the same latent as the target branch
- LoRA is a **parallel residual path on `x`**
- the style term is a **subspace penalty**, not just another direct teacher-matching loss

These are not cosmetic distinctions; they are structurally important to the method actually implemented here.

## Relation to ESD-x, UCE, and Concept Ablation

SPACE is closest in spirit to ESD-x because it performs a targeted edit on cross-attention behavior, but it differs by using a richer teacher target, a dual-anchor construction, trajectory-band supervision, vulnerable-style protection, and preserve replay. Compared with UCE, SPACE is not a closed-form edit over erased and preserved concepts; it is an iterative teacher-student optimization over diffusion predictions with explicit replay terms. Compared with Concept Ablation, SPACE does make use of a neutral art-like anchor, but it does not stop at anchor imitation; it explicitly subtracts the target style residual and regularizes preservation along several axes. The intended output is therefore not replacement with another artist, but **neutralized style suppression with preserved content and utility**.

## Claims, Limits, and Current Scope

This README describes the SPACE implementation in [training/space_sd.py](../training/space_sd.py) and [core/space_trainer.py](../core/space_trainer.py). The scope of the claim should remain careful: SPACE targets output-level artist/style suppression with preservation under the current evaluation harness. It should not be described as proof of complete deletion of all target knowledge from the model weights, nor as a legal claim of perfect machine unlearning in the strongest possible sense.
