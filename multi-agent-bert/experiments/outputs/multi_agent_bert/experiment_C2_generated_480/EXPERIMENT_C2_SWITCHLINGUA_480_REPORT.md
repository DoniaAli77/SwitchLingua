# Experiment C2 — SwitchLingua Generated 480 (primary_only)

New experiment, **separate** from Exp A (EESA-trained), Exp C (240 generated), and
Exp D (Ahmed models). Fine-tune XLM-R on **480 generated** sentiment samples,
evaluate primary_only on the real EESA test (818). Date: 2026-06-19.

## Dataset (verified)
`data/Sentiment/generated/merged/switchlingua_sentiment_train_480_160perlabel.jsonl`
- **480 rows**, perfectly balanced: **160 positive / 160 negative / 160 neutral**
- 0 empty texts; 0 exact duplicates; 0 normalized duplicates
- labels exactly {positive, negative, neutral}; fields `text`, `label`

## Training setup (same recipe as Exp C — controlled for a fair 240→480 comparison)
| param | value |
|---|---|
| base | xlm-roberta-base (fine-tune) |
| optimizer | adafactor |
| lr / epochs | 2e-5 / 4 |
| batch_size × grad_accum | 4 × 4 (eff. 16) |
| max_length / seed | 256 / 42 |
| fp16 / grad-checkpointing | on / on |
| dev (validation) | EESA dev (818) |
| train_runtime / final_train_loss | 340 s / 0.6113 |

Dev (EESA dev) by epoch: 0.561 → 0.590 → **0.665** → 0.658 (saved = final epoch 4:
acc 0.6577 / macro F1 0.6484).

## Primary_only result — EESA test (818)
**accuracy 0.6491 · macro F1 0.6382 · weighted F1 0.6553**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.828 | 0.661 | **0.735** | 363 |
| negative | 0.554 | 0.599 | **0.576** | 197 |
| neutral | 0.549 | 0.671 | **0.604** | 258 |

Confusion matrix (rows = true, cols = predicted; order pos / neg / neu):
```
            pred_pos  pred_neg  pred_neu
true_pos       240       42        81
true_neg        18      118        61
true_neu        32       53       173
```
Prediction distribution: **positive 290 · negative 213 · neutral 315**
(true: positive 363 · negative 197 · neutral 258) — fairly balanced, slight
over-prediction of neutral.

**Checkpoint:** `experiments/checkpoints/expC2_switchlingua_xlmr_480/`

## Comparison
### vs Experiment C (240 generated) — FAIR (identical recipe, only data size differs)
| metric | C (240) | **C2 (480)** | Δ |
|---|---|---|---|
| accuracy | 0.5905 | **0.6491** | **+0.0586** |
| macro F1 | 0.5619 | **0.6382** | **+0.0763** |
| positive F1 | 0.703 | 0.735 | +0.032 |
| negative F1 | 0.510 | 0.576 | +0.066 |
| neutral F1 | 0.473 | 0.604 | **+0.131** |

**Doubling the generated data (240→480) gives a clear, controlled gain** (+5.9 acc /
+7.6 macro F1). Biggest improvements on the previously-weak **neutral** and
**negative** classes; predictions are much better balanced (Exp C leaked 95
true-neutral → positive; C2 only 32).

### vs Experiment A (2,464 real EESA) — reference, NOT controlled
| | Exp A (real, AdamW) | C2 (generated, Adafactor) |
|---|---|---|
| accuracy | 0.8240 | 0.6491 |
| macro F1 | 0.8088 | 0.6382 |

Still below the real-data model (expected: 5× less data and AraBERT-vs-generated
domain gap; also AdamW vs Adafactor). C2 narrows the gap vs Exp C but does not close
it.

## Notes / caveats
- This is **primary_only**; `full_agentic` not run yet (per instruction — wait to
  see primary_only first).
- C2 vs C is a clean "more generated data" comparison (same recipe). C2 vs A still
  carries the optimizer confound (Adafactor vs AdamW).
- Architecture unchanged; no Ahmed models involved; outputs isolated under
  `experiment_C2_generated_480/`.
