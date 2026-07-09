# Thomas Kinkade metric validation

**Assessment: PASS WITH CAVEATS**

## Independently checked FID

- Canonical `torch-fidelity` FID using native file preprocessing: **14.578488**
- Prior TorchMetrics FID after explicit 299×299 bilinear resize: **14.089452**
- Absolute implementation difference: **0.489036**
- Inputs: 30,000 unique official COCO JPGs and 30,000 exactly ID-matched SPACE PNGs
- Safety-filtered black images included: 144 (0.48%)

The canonical `torch-fidelity` value is the repository-comparable result. The prior value remains documented as a separate preprocessing implementation and must not be silently mixed with it.

## Paired metrics

- Exactly 500 matched vanilla/SPACE pairs were aggregated.
- CLIP-drop arithmetic identity: PASS
- Style-drop arithmetic identity: PASS
- All reported paired metrics finite: PASS
- Style target hits: 242/500

## Required interpretation caveats

1. Published SD v1.4 FID 14.50 is context only unless every generation and preprocessing setting matches.
2. The 0.48% black-image rate is part of the evaluated saved pipeline output and can affect FID.
3. Style target rate is a custom five-label CLIP diagnostic, not a standardized benchmark.
4. Paired metrics use 500 matched prompt outputs; FID uses a separate 30,000-image COCO evaluation.
