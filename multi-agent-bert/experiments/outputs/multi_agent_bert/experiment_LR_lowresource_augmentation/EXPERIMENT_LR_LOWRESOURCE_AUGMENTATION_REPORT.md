# Experiment LR — Low-Resource Augmentation (EESA% ± SwitchLingua-960)

Does generic SwitchLingua data help when real labeled data is scarce? Tests EESA
10/25/50% with vs without the 960 generated samples. Date: 2026-06-24.

## Design (matched compute, not matched epochs)
"Same epochs" was abandoned because it starved the small subsets (10% × 4 epochs =
64 steps → the EESA-only baseline collapsed to all-positive, acc 0.444 / macroF1
0.205). Instead, **matched optimization steps**:
- Fresh `xlm-roberta-base`, adafactor, lr 2e-5, batch 4 × grad_accum 4, max_length
  256, fp16, seed 42.
- **`--max_steps 400`** (same compute budget for every run) + **`--load_best`**
  (keep best-dev checkpoint by macro F1, eval/save every 50 steps).
- Stratified EESA subsets (preserve the pos-heavy prior) from the leak-cleaned train.
- Dev = EESA dev; test = EESA test. primary_only only.

## Results — EESA test (818)
| EESA real | real n | gen % of mix | only: acc / macroF1 | +gen960: acc / macroF1 | **Δ acc / Δ macroF1** |
|---|---|---|---|---|---|
| 10% | 246 | 80% | 0.7751 / 0.7586 | 0.7408 / 0.7295 | **−0.0342 / −0.0291** |
| 25% | 615 | 61% | 0.7873 / 0.7758 | 0.7689 / 0.7545 | **−0.0183 / −0.0213** |
| 50% | 1232 | 44% | 0.8166 / 0.8025 | 0.8142 / 0.8008 | **−0.0024 / −0.0017** |
| 100%* | 2463 | 28% | 0.8533 / 0.8409 | 0.8411 / 0.8294 | −0.0122 / −0.0115 |

\*100% row = earlier E0 vs E3 (4-epoch recipe, single seed); shown for context only.
Per-class F1 and prediction distributions confirm the augmented models drift toward
the generated distribution (lower neutral F1, more balanced predictions).

## Findings
1. **Generated augmentation never helped — every Δ is negative**, across all data
   sizes.
2. **The harm is monotone in the generated fraction of the mix** (matched runs):
   −0.034 at 80% gen → −0.018 at 61% → −0.002 at 44% (≈ neutral).
3. **It hurts *most* when real data is scarcest** — the opposite of the usual
   low-resource augmentation expectation. With only 246 real samples, the 960
   generated are **80%** of training, so the model mostly fits the generated
   (off-domain) distribution and lands near the generated-only ceiling (~0.74)
   instead of the real-EESA distribution.
4. **Real-only scales cleanly**: 0.775 (10%) → 0.787 (25%) → 0.817 (50%) → 0.853
   (100%) — more real data is what helps.

## Interpretation
This confirms the **domain-compatibility** explanation (see
`EXPERIMENT_E_AUGMENTATION_FAILURE_DIAGNOSIS.md`): generated and EESA are different
registers, so mixing generated data pulls the model off-target, **proportional to
how much it dominates the training mix**. Scarce real data makes this worse, not
better, because the fixed 960-sample generated set then swamps the small real set.

**Answer to the research question:** generic SwitchLingua data does **not** help as a
naive large-ratio augmenter of a different-domain real set, even (especially) in
low-resource conditions. Its value remains as **standalone** training data (its own
domain, or where no real target data exists; C1–C3 + agent rescue), not as
augmentation of EESA. If augmentation is pursued, the **generated:real ratio must be
kept small** (downsample generated / upweight real) — at 50% real (44% gen) the
effect was already negligible.

## Caveats
- **Single seed per cell** — the 10→25→50% *trend* is clean and monotone, but for
  firm per-point claims a ≥3-seed repeat is warranted (seed std ~±0.02 measured
  earlier).
- The 100% row used a different (4-epoch) recipe, so compare it only loosely with
  the matched-steps 10/25/50% rows.
- Only one ratio tested (fixed 960 generated). A **ratio sweep** (e.g., 1×/0.5×/0.25×
  generated relative to real) would map where augmentation becomes neutral/positive.
