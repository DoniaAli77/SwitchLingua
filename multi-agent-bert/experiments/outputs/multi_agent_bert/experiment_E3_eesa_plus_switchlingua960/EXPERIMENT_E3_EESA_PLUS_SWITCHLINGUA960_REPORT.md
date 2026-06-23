# Experiment E3 — EESA + SwitchLingua-960 Augmentation (primary_only)

Augmentation experiment (separate from generated-only C1/C2/C3). Fresh
`xlm-roberta-base` fine-tuned on **real EESA train + 960 generated** samples;
evaluate primary_only on real EESA test. Date: 2026-06-23.

## Data prep / merge (steps 1–6)
| set | rows | labels (pos / neg / neu) |
|---|---|---|
| EESA train (real) | 2,464 | 1,092 / 594 / 778 (imbalanced) |
| Generated-960 | 960 | 320 / 320 / 320 (balanced) |
| **Merged (after leakage removal)** | **3,423** | 1,411 / 914 / 1,098 |

- **Overlap EESA-train ∩ generated: 0** (no duplicate augmentation).
- EESA-train internal dups: 2 · generated internal dups: 0.
- **Leakage check:** 1 train↔dev overlap found — it was a **pre-existing EESA
  train/dev duplicate** (`'اغنية top'`), *not* from the generated data. Removed from
  train → merged = 3,423 (2,463 EESA + 960 gen). **Post-clean leakage: dev 0, test 0.**
- Labels exactly {positive, negative, neutral}; 0 empty texts.
- Merged file: `data/Sentiment/processed/augmentation/eesa_train_plus_switchlingua_960.jsonl`.

## Training settings
`xlm-roberta-base` (fresh; not continued from EESA or C3) · adafactor · lr 2e-5 ·
4 epochs · batch 4 × grad_accum 4 (eff 16) · max_length 256 · fp16 ·
gradient_checkpointing · seed 42 · save_steps 400.
Checkpoint: `experiments/checkpoints/expE3_eesa_plus_switchlingua960_xlmr/`.

**Dev (EESA dev) by epoch:** 0.798 → 0.809 → 0.830 → **0.840** (final
acc 0.8399 / macro F1 0.8292) — best dev.

## Primary_only result — EESA test (818)
**accuracy 0.8411 · macro F1 0.8294 · weighted F1 0.8404**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.887 | 0.909 | 0.898 | 363 |
| negative | 0.782 | 0.817 | 0.799 | 197 |
| neutral | 0.821 | 0.764 | 0.791 | 258 |

Confusion (rows = true, cols = predicted; pos / neg / neu):
```
            pred_pos  pred_neg  pred_neu
true_pos       330       13        20
true_neg        13      161        23
true_neu        29       32       197
```
Prediction distribution: positive 372 · negative 206 · neutral 240
(true 363 / 197 / 258 — well calibrated).

## Comparison (EESA test, primary_only unless noted)
| system | accuracy | macro F1 |
|---|---|---|
| Generated-only C3-960 (3-seed mean) | 0.6695 | 0.6592 |
| Generated-only C3-960 best-dev **full_agentic** | 0.7543 | 0.7387 |
| EESA-only XLM-R (reference, **AdamW**) | 0.8240 | 0.8088 |
| **E3: EESA + generated-960 (Adafactor)** | **0.8411** | **0.8294** |

**E3 beats the EESA-only baseline by +0.0171 acc / +0.0206 macro F1** and is the
strongest sentiment **primary** so far — gains spread across all classes
(esp. negative/neutral, the EESA-imbalanced minority classes).

## ⚠️ Important caveat (clean-comparison control needed)
E3 uses **Adafactor**; the EESA-only 0.8240 reference used **AdamW**. So the +0.017
gain mixes two factors — the **augmentation** *and* the **optimizer change**. To
attribute the gain to augmentation cleanly, we need an **EESA-only + Adafactor**
control (same recipe, no generated data). The dev signal (0.840 vs EESA-only test
0.824) is encouraging, but **the matched control is required before claiming the
generated data helps as augmentation.** Recommended as the immediate next run
(~30 min, no API cost).

## Notes
- primary_only only; **full_agentic not run** (per instruction).
- Fresh from `xlm-roberta-base`; not continued from EESA or C3; no Ahmed models.
- Outputs isolated under `experiment_E3_eesa_plus_switchlingua960/`; C1/C2/C3/D and
  topic outputs untouched.
