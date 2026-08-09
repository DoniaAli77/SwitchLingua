# ArabicDominant-180 vs Original-180 - matched-compute ablation at 136 steps

Two FIXED balanced corpora, model seeds 42/43/44, exactly 136 optimizer steps each.
Identical preprocessing, hyperparameters, label mapping, validation procedure (no dev),
checkpoint selection (final step, load_best off) and Silver-1163 primary_only evaluation.
The training corpus is the only difference. No data was generated for this run.

Collapse criterion: fewer than 9 predicted classes, or a top predicted-class share >= 50%. (Healthy Topic-540 reference: 9 classes, top share ~32%.)

## Per-run, Silver-1163

| arm | seed | acc | macro-F1 | weighted-F1 | #pred classes | top class | top share | collapsed |
|---|--:|--:|--:|--:|--:|---|--:|---|
| Original-180 | 42 | 0.6148 | 0.5476 | 0.5914 | 9 | tech | 34.5% | no |
| Original-180 | 43 | 0.6148 | 0.5468 | 0.6001 | 9 | tech | 32.8% | no |
| Original-180 | 44 | 0.6320 | 0.5680 | 0.6167 | 9 | tech | 27.6% | no |
| ArabicDominant-180 | 42 | 0.6062 | 0.5294 | 0.5693 | 9 | tech | 35.0% | no |
| ArabicDominant-180 | 43 | 0.6036 | 0.5372 | 0.5766 | 9 | tech | 29.2% | no |
| ArabicDominant-180 | 44 | 0.6096 | 0.5414 | 0.5842 | 9 | tech | 31.9% | no |

## Mean +/- sd over the three model seeds

| arm | acc | macro-F1 | weighted-F1 |
|---|---|---|---|
| Original-180 | 0.6205 +/- 0.0081 | 0.5541 +/- 0.0098 | 0.6028 +/- 0.0105 |
| ArabicDominant-180 | 0.6065 +/- 0.0025 | 0.5360 +/- 0.0050 | 0.5767 +/- 0.0060 |

## Paired differences (ArabicDominant - Original), same model seed

| seed | d acc | d macro-F1 | d weighted-F1 |
|--:|--:|--:|--:|
| 42 | -0.0086 | -0.0182 | -0.0221 |
| 43 | -0.0112 | -0.0096 | -0.0235 |
| 44 | -0.0224 | -0.0266 | -0.0326 |
| **mean** | **-0.0140** | **-0.0181** | **-0.0261** |

## Per-class F1 (mean over seeds)

| class | Original-180 | ArabicDominant-180 | delta |
|---|--:|--:|--:|
| business | 0.396 | 0.249 | -0.147 |
| education | 0.480 | 0.422 | -0.057 |
| health | 0.399 | 0.447 | +0.048 |
| shopping | 0.540 | 0.573 | +0.034 |
| medical | 0.588 | 0.618 | +0.030 |
| sports | 0.602 | 0.552 | -0.050 |
| tech | 0.733 | 0.716 | -0.017 |
| finance | 0.773 | 0.748 | -0.025 |
| social | 0.476 | 0.497 | +0.022 |

## Interpretation (rule applied mechanically)

- collapsed runs: NONE
- acc: -0.0086 -0.0112 -0.0224 -> consistent
- macro_f1: -0.0182 -0.0096 -0.0266 -> consistent
- weighted_f1: -0.0221 -0.0235 -0.0326 -> consistent

**VERDICT: directional evidence AGAINST Arabic-ratio alignment.** All three paired differences share a direction on both accuracy and macro-F1, and no run collapsed. Three seeds; no significance claimed.

These 136-step results are NOT combined with the 48-step diagnostic, which is an
undertraining artefact and does not demonstrate a cs_ratio effect.