# ArabicDominant-180 vs Original-180 — exploratory configuration ablation

Single factor: generation cs_ratio [50%,60%] -> [85%,95%]. Identical training recipe
(xlm-roberta-base, 4 epochs = 48 steps, batch 16, lr 2e-5, maxlen 256, fp16, adamw_torch,
load_best off, no dev), identical preprocessing, seeds 42/43/44, primary_only evaluation
on the unchanged Silver-1163. Exploratory: 3 seeds, no significance test.

## Per-run, Silver-1163

| arm | seed | acc | macro-F1 | weighted-F1 |
|---|--:|--:|--:|--:|
| Original-180 | 42 | 0.1668 | 0.1810 | 0.1509 |
| Original-180 | 43 | 0.0705 | 0.0391 | 0.0270 |
| Original-180 | 44 | 0.1694 | 0.1106 | 0.1539 |
| ArabicDominant-180 | 42 | 0.2279 | 0.1884 | 0.1927 |
| ArabicDominant-180 | 43 | 0.0610 | 0.0128 | 0.0070 |
| ArabicDominant-180 | 44 | 0.4205 | 0.3595 | 0.4080 |

## Arms reported separately

| arm | set | acc | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| Original-180 | Silver-1163 | 0.1356 ± 0.0460 | 0.1102 ± 0.0580 | 0.1106 ± 0.0591 |
| Original-180 | Silver-1044 | 0.1315 ± 0.0443 | 0.1060 ± 0.0542 | 0.1046 ± 0.0560 |
| ArabicDominant-180 | Silver-1163 | 0.2365 ± 0.1469 | 0.1869 ± 0.1415 | 0.2026 ± 0.1638 |
| ArabicDominant-180 | Silver-1044 | 0.2391 ± 0.1526 | 0.1883 ± 0.1443 | 0.2076 ± 0.1715 |

## Paired per-seed deltas (ArabicDominant - Original), Silver-1163

| seed | Δ acc | Δ macro-F1 | Δ weighted-F1 |
|--:|--:|--:|--:|
| 42 | +0.0610 | +0.0073 | +0.0418 |
| 43 | -0.0095 | -0.0263 | -0.0200 |
| 44 | +0.2511 | +0.2489 | +0.2541 |
| **mean** | **+0.1009** | **+0.0767** | |

Direction consistent across all three seeds: NO (accuracy), NO (macro-F1).

## Per-class F1, mean over seeds, Silver-1163

| class | Original-180 | ArabicDominant-180 | Δ |
|---|--:|--:|--:|
| business | 0.140 | 0.091 | -0.050 |
| education | 0.050 | 0.085 | +0.035 |
| health | 0.068 | 0.211 | +0.143 |
| shopping | 0.129 | 0.144 | +0.015 |
| medical | 0.087 | 0.117 | +0.030 |
| sports | 0.238 | 0.454 | +0.216 |
| tech | 0.136 | 0.173 | +0.037 |
| finance | 0.080 | 0.408 | +0.328 |
| social | 0.064 | 0.000 | -0.064 |

## Caveats

- The Original arm uses a different nested 180-subset per seed, so its spread includes
  subset-sampling variance; the ArabicDominant arm has one fixed 180 corpus (205 accepted
  sentences total), so its spread is training variance only.
- 3 seeds, one dataset per arm: exploratory, no significance test.
- Silver is final evaluation only; it never informed generation, training or selection.