> **⚠️ UPDATE (seed-stability check):** the "480 > 960 / plateau-reversal" finding
> below is a **single-seed artifact and is RETRACTED**. Across 3 seeds, 960 averages
> **higher** than 480 (0.6695 vs 0.6500 acc); this C3 seed-42 run was the low
> outlier of the 960 group. See
> `experiment_seed_stability/EXPERIMENT_SEED_STABILITY_C2_C3_REPORT.md`.

# Experiment C3 — SwitchLingua Generated 960 Sentiment (primary_only)

Separate experiment (not mixed with C1/C2/D or topic T1/T2). Fresh
`xlm-roberta-base` fine-tune on 960 generated sentiment samples; evaluate
primary_only on real EESA test. Date: 2026-06-21.

## 1. Dataset sizes
| split | rows | source |
|---|---|---|
| train | 960 | `data/Sentiment/generated/merged/switchlingua_sentiment_train_960_320perlabel.jsonl` |
| dev | 818 | `data/Sentiment/processed/eesa_sentiment_dev.jsonl` (real EESA) |
| test | 818 | `data/Sentiment/processed/eesa_sentiment_test.jsonl` (real EESA) |

## 2. Label distribution (train)
Balanced: **320 positive / 320 negative / 320 neutral**. 0 empty, 0 duplicates,
labels exactly {positive, negative, neutral}.

## 3. Training settings (identical to C1/C2 — fair scaling comparison)
`xlm-roberta-base` (fresh, not continued from 240/480) · adafactor · lr 2e-5 ·
4 epochs · batch 4 × grad_accum 4 (eff 16) · max_length 256 · fp16 ·
gradient_checkpointing · seed 42.

## 4. Best dev metric (EESA dev)
By epoch: 0.6443 → 0.6443 → **0.6589** → 0.6455. Saved = final epoch 4 =
**acc 0.6455 / macro F1 0.6398 / weighted F1 0.6535**.

## 5–10. Primary_only result — EESA test (818)
**accuracy 0.6381 · macro F1 0.6322 · weighted F1 0.6454**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.844 | 0.612 | 0.709 | 363 |
| negative | 0.504 | 0.706 | 0.588 | 197 |
| neutral | 0.577 | 0.624 | 0.600 | 258 |

Confusion (rows = true, cols = predicted; pos / neg / neu):
```
            pred_pos  pred_neg  pred_neu
true_pos       222       72        69
true_neg         9      139        49
true_neu        32       65       161
```
Prediction distribution: **positive 263 · negative 276 · neutral 279**
(true 363 / 197 / 258).

## 11. Comparison — generated-data scaling (same recipe) + reference
| model | train | accuracy | macro F1 |
|---|---|---|---|
| C1 generated-240 | 240 | 0.5905 | 0.5619 |
| **C2 generated-480** | 480 | **0.6491** | **0.6382** |
| C3 generated-960 | 960 | 0.6381 | 0.6322 |
| Exp A real-EESA (AdamW) — reference | 2,464 | 0.8240 | 0.8088 |

**Key finding — the gain plateaus and slightly reverses at 960:**
- 240 → 480: **+0.059 acc / +0.076 macro F1** (clear gain).
- 480 → 960: **−0.011 acc / −0.006 macro F1** (slight regression, not improvement).

So doubling from 480 to 960 generated samples did **not** help EESA transfer — it
plateaued around **~0.64–0.65 accuracy** and dipped slightly. The dev curve agrees
(C3 dev 0.6455 < C2 dev ~0.658), so it's a genuine convergence point, not just test
noise. The regression is driven by a **distribution shift toward negative**: C3
predicts 276 negatives (vs C2's 213), dropping positive recall to 0.612 (C2 ~0.755)
— 72 true-positives leak to negative.

**Interpretation:** generated-data transfer to real EESA appears to hit a ceiling
(domain gap between generated and real text) around ~480 samples; more generated
data beyond that doesn't close the gap to the 2,464-sample real-data model and can
mildly shift the decision boundary. (Caveat: single run per size — some of the
480-vs-960 gap could be optimization variance; a seed sweep would confirm.)

## 12. Notes
- primary_only only; **full_agentic NOT run** (stopped for review, per instruction).
- C1/C2/D and topic outputs untouched; C3 isolated under
  `experiment_C3_generated_960/`. Checkpoint:
  `experiments/checkpoints/expC3_switchlingua_xlmr_960/`.
