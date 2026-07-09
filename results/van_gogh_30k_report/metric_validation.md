# Van Gogh metric validation

**Assessment: PASS WITH CAVEATS**

## Canonical FID

- Canonical `torch-fidelity` FID (native preprocessing, 30,000 images vs. official COCO): **14.567865**
- Inputs: 30,000 unique official COCO JPGs and 30,000 exactly ID-matched SPACE PNGs
- Safety-filtered black images included: 162 (0.54%)

## Paired metrics

- Exactly 500 matched vanilla/SPACE pairs were aggregated.
- CLIP-drop arithmetic identity: PASS
- Style-drop arithmetic identity: PASS
- Style target hits: 162/500

## Required interpretation caveats

1. Published SD v1.4 FID 14.50 is context only unless every generation and preprocessing setting matches.
2. The black-image rate is part of the evaluated saved pipeline output and can affect FID.
3. Style target rate is a custom five-label CLIP diagnostic, not a standardized benchmark.
4. Paired metrics use 500 matched prompt outputs; FID uses a separate 30,000-image COCO evaluation.
