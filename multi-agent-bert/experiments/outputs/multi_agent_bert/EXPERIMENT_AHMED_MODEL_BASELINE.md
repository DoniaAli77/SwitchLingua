# Experiment — Ahmed Model: External EESA Baseline

Ahmed's **provided test predictions were evaluated on the EESA test set.** We did
**not** retrain, regenerate, or reproduce Ahmed's preprocessing/feature pipeline from
raw text — this report scores the prediction arrays Ahmed supplied. Date: 2026-06-27.

## Input files used
`ahmed models/`: `ArEnSAData-preFinal-NDCS3C-test.csv` (Ahmed's exact test file: no
header, col 0 = text, col 1 = true_label), `y_true.npy`, `y_pred.npy`,
`y_pred_prob.npy` (+ `script.py.txt`).

## Label mapping (confirmed by Ahmed)
`tag2idx = {negative:0, neutral:1, positive:2}` →
`y_pred_prob` columns = **[0]=prob_negative, [1]=prob_neutral, [2]=prob_positive**.
(`y_true`/`y_pred` are stored as label **strings**.)

## Shape + probability validation
| check | result |
|---|---|
| `y_true` shape | (818,) ✅ |
| `y_pred` shape | (818,) ✅ |
| `y_pred_prob` shape | (818, 3) ✅ |
| `argmax(y_pred_prob)` → tag **== `y_pred`** | **818/818** ✅ |
| probability rows sum to 1 | ✅ (min 1.0000, max 1.0000) |
| `y_true` label distribution | negative 197 · neutral 258 · positive 363 |
| **matches our EESA test distribution** (197/258/363) | ✅ **same test set** |

**Alignment (resolved).** After receiving Ahmed's exact test CSV
(`ArEnSAData-preFinal-NDCS3C-test.csv`), the prediction rows were aligned to text. The
CSV labels match `y_true.npy` for **all 818/818 samples** (same order), so each row's
text, true label, prediction, and probabilities are jointly aligned. A sample-level
aligned file is provided and is **the basis for any future sample-level analysis**:
`data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv`
(columns: `sample_id, text, true_label, pred_label, prob_negative, prob_neutral,
prob_positive, confidence`; `pred_label == argmax(prob)` for 818/818).
*(Note: Ahmed's test set is the same 818 EESA-test samples as ours by label
distribution and text; its row order differs from our `eesa_sentiment_test.jsonl`,
which is immaterial now that we align on Ahmed's own text.)*

## Results — EESA test (818)
**accuracy 0.9254 · macro F1 0.9207 · weighted F1 0.9254** (reproduces Ahmed's
reported 0.9254 / 0.9207 / 0.9254 exactly).

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.9482 | 0.9587 | 0.9534 | 363 |
| negative | 0.9409 | 0.8883 | 0.9138 | 197 |
| neutral | 0.8830 | 0.9070 | 0.8948 | 258 |

Confusion matrix (rows = true, cols = predicted; pos / neg / neu):
```
            pred_pos  pred_neg  pred_neu
true_pos       348        3        12
true_neg         3      175        19
true_neu        16        8       234
```
Prediction distribution: positive 367 · negative 186 · neutral 265
(true 363 / 197 / 258 — well calibrated).

## Comparison with our models (EESA test, primary_only unless noted)
| system | accuracy | macro F1 | notes |
|---|---|---|---|
| **Ahmed model (external)** | **0.9254** | **0.9207** | char-CNN+BiLSTM+AraBERT features; provided predictions |
| EESA-only XLM-R, Adafactor (E0) | 0.8533 | 0.8409 | our best primary |
| EESA full_agentic reference (Exp A) | 0.8509 | 0.8401 | XLM-R primary + LLM agents |
| EESA-only XLM-R, AdamW (ref) | 0.8240 | 0.8088 | original reference |
| Generated-only C3 primary (3-seed mean) | 0.6695 | 0.6592 | 960 generated only |
| Generated-only C3 full_agentic (seed 456) | 0.7543 | 0.7387 | generated + LLM agents |

Augmentation results (E3/LR/ratio) are not directly comparable here (they study
generated-as-augmentation, not a standalone EESA model) and are omitted.

**Ahmed's model is the strongest EESA sentiment model on record**: **+0.072 accuracy /
+0.080 macro F1 over our best XLM-R primary (E0)**, and +0.075 acc over the XLM-R
full_agentic reference. Its per-class F1 ≥ 0.89 across all classes, with the main
residual confusion being neutral↔negative (19 + 8).

## Caveat
Ahmed's model is **evaluated from provided predictions, not locally reproduced from
raw text.** We did not run Ahmed's TensorFlow/Keras model or rebuild its
AraBERT-feature + char-vocab preprocessing (those artifacts were not available; see
`experiment_D/`). The numbers above are a faithful scoring of Ahmed's supplied
`y_true`/`y_pred`/`y_pred_prob` on the EESA test set.

---

## Optional next step — Ahmed-as-frozen-primary experiment (proposal only — NOT run)
Now that the **text-aligned probabilities** exist
(`ahmed_eesa_test_predictions_aligned.csv`), Ahmed's model can be wired in as a
**frozen primary source** in the router / full_agentic pipeline:

1. **Adapter:** a `PrecomputedPrimaryClassifier` reads the aligned CSV and, keyed by
   `text` (normalized), emits `ModelOutput(label=pred_label,
   confidence=confidence, probabilities={negative,neutral,positive})` — exactly the
   interface the router expects. No TF/Keras in the pipeline.
2. **primary_only:** equals Ahmed's standalone result directly — **0.9254 / 0.9207**.
3. **frozen-primary full_agentic:** the router escalates only low-confidence Ahmed
   predictions (threshold 0.9) to the LLM agents (Fix-2 consensus on); compare
   **Ahmed primary_only vs Ahmed frozen-primary full_agentic.**
4. **Expectation: little or no improvement.** At 0.9254 the primary is already
   near-perfect, so — like the topic experiment (0.99 primary → agents net-neutral) —
   escalation touches few samples and agents are unlikely to help (may even add noise).
   The value is the controlled test, not an expected gain.
5. **Scope:** Ahmed provided **test** predictions only (no train/dev) → inference-time
   integration (frozen primary), not retraining. Standalone caveat (below) still holds.

**Run only if approved.**
