# Experiment E0 — EESA-only Adafactor Control (primary_only)

The matched control for E3: same recipe, EESA-only (no generated data). Isolates
the augmentation effect from the optimizer change. Date: 2026-06-23.

## 1–3. Data
| set | rows | labels (pos / neg / neu) |
|---|---|---|
| train (EESA-only, control) | **2,463** | 1,091 / 594 / 778 |
| dev (EESA) | 818 | — |
| test (EESA) | 818 | — |

**Same train↔dev duplicate removed** as in E3: `'اغنية top'` (positive) — confirmed
1 row removed (2,464 → 2,463). Post-clean leakage: dev 0, test 0. This train set is
**exactly E3's training data minus the 960 generated samples** → clean control.

## 4. Training settings (identical to E3)
`xlm-roberta-base` (fresh) · adafactor · lr 2e-5 · 4 epochs · batch 4 × grad_accum 4
(eff 16) · max_length 256 · fp16 · gradient_checkpointing · seed 42 · save_steps 400.
Checkpoint: `experiments/checkpoints/expE0_eesa_only_adafactor_xlmr/`.

## 5. Dev (EESA dev) by epoch
0.736 → 0.830 → 0.839 → **0.851** (final acc 0.8509 / macro F1 0.8400).

## 6–11. Primary_only — EESA test (818)
**accuracy 0.8533 · macro F1 0.8409 · weighted F1 0.8530**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.907 | 0.917 | 0.912 | 363 |
| negative | 0.835 | 0.772 | 0.802 | 197 |
| neutral | 0.792 | 0.826 | 0.808 | 258 |

Confusion (rows = true, cols = predicted; pos / neg / neu):
```
            pred_pos  pred_neg  pred_neu
true_pos       333        7        23
true_neg        12      152        33
true_neu        22       23       213
```
Prediction distribution: positive 367 · negative 182 · neutral 269.

## 12. Comparison + effect decomposition (EESA test, primary_only)
| system | optimizer | gen data | accuracy | macro F1 |
|---|---|---|---|---|
| EESA-only reference | AdamW | no | 0.8240 | 0.8088 |
| **E0: EESA-only** | **Adafactor** | no | **0.8533** | **0.8409** |
| E3: EESA + generated-960 | Adafactor | yes | 0.8411 | 0.8294 |

**Decomposition:**
- **Optimizer effect** (AdamW → Adafactor, EESA-only): **+0.0293 acc / +0.0321 macro F1**
  (0.8240 → 0.8533).
- **Augmentation effect** (E0 → E3, both Adafactor): **−0.0122 acc / −0.0115 macro F1**
  (0.8533 → 0.8411).

## Conclusion — augmentation does NOT help here
- **E3's apparent +0.017 "gain" over the old baseline was the optimizer, not the
  generated data.** Properly controlled (E0 vs E3, both Adafactor), adding 960
  generated samples **did not help — it slightly lowered** accuracy/macro F1.
- **Caveat (important):** the augmentation gap (−0.012 acc) is **within the seed
  variance** measured earlier (~±0.02), so the honest statement is *"no benefit,
  slight non-significant negative trend"* — not a firm "hurts." A multi-seed E0-vs-E3
  comparison (≥3 seeds each) would be needed to confirm sign and magnitude.
- Likely mechanism: the balanced generated data (320/320/320) plus its
  generated-vs-real domain gap pulls the model off the real EESA distribution
  (pos-heavy), slightly degrading rather than augmenting.

**Net:** SwitchLingua-generated data is strong as a *standalone* training source
(C1–C3: 0.59 → 0.67, agents lift to 0.75) but, as **augmentation on top of real
EESA**, it provides **no measurable benefit** at 960 samples — the real-data model
(Adafactor) already reaches 0.853 and the generated data does not add to it.

Stopped after primary_only. No full_agentic; no Ahmed models; E3 outputs untouched.
