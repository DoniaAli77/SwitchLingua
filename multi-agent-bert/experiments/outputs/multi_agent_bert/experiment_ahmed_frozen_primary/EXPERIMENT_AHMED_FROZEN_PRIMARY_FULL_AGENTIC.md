# Experiment — Ahmed Frozen-Primary + Full-Agentic Layer

**Ahmed's provided aligned predictions were used as a frozen primary classifier, and
the agentic layer was evaluated on top of them.** No training, no generation, no
modification of Ahmed's predictions; this does **not** reproduce Ahmed's
preprocessing/training pipeline. Date: 2026-06-27.

## Setup
- Frozen primary = `PrecomputedPrimaryClassifier` reading
  `data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv`
  (keyed by `sample_id`) → `ModelOutput(label=pred_label, confidence=confidence,
  probabilities={negative,neutral,positive})`. No TF/Keras in the pipeline.
- active_task = sentiment_classification · pipeline_mode = full_agentic ·
  **threshold = 0.7** · Fix-2 primary-aware consensus ON (w_primary = 1.0) ·
  agents_use_primary_signal = false · LLM = GPT-4o-mini. EESA test (818).
- Integration validated: **primary_only through the pipeline reproduces Ahmed's
  baseline exactly (0.9254 / 0.9207).**

## 1. Ahmed primary_only (frozen primary, via pipeline)
accuracy **0.9254** · macro F1 **0.9207** · weighted F1 **0.9254**.

## 2. Ahmed full_agentic, threshold 0.7
accuracy **0.9205** · macro F1 **0.9153** · weighted F1 **0.9202** (0 connection /
0 quota errors).

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.941 | 0.964 | 0.952 | 363 |
| negative | 0.922 | 0.904 | 0.913 | 197 |
| neutral | 0.889 | 0.872 | 0.881 | 258 |

Confusion (rows = true, cols = predicted; pos / neg / neu):
```
            pred_pos  pred_neg  pred_neu
true_pos       350        2        11
true_neg         2      178        17
true_neu        20       13       225
```
Prediction distribution: positive 372 · negative 193 · neutral 253.

## 3–7. Escalation + agent effect
| metric | value |
|---|---|
| escalated | **84 / 818 (10.3%)** |
| escalated-only accuracy (full_agentic) | 0.702 |
| **wrong→correct** | 11 |
| **correct→wrong** | 15 |
| **net change** | **−4** |

The agents fixed 11 and broke 15 of the 84 escalated (genuinely uncertain) samples →
**net −4** → exactly the −0.0049 overall accuracy drop (0.9254 → 0.9205).

## 8. Cost
**~$0.043** (84 escalated × 4 calls ≈ **336 LLM calls**, gpt-4o-mini).

## 9. Comparison vs the aborted threshold=0.9 attempt
| run | threshold | escalated | result |
|---|---|---|---|
| Ahmed primary_only | — | — | 0.9254 / 0.9207 |
| Ahmed full_agentic | **0.7 (calibrated)** | 84 (10.3%) | **0.9205 / 0.9153** (Δ −0.0049) |
| Ahmed full_agentic | 0.9 (aborted) | **818 (100%)** | invalid — see §10 |

The threshold-0.9 attempt escalated **all 818 samples** and was aborted (heavy
rate-limiting; ~3.5 h projected). It is **not a valid agentic test** for this primary.

## 10. Why threshold 0.9 is invalid/unfair for Ahmed
Ahmed's softmax is **not peaked**: confidence ranges min 0.389 / median 0.788 /
**max 0.864**. No sample reaches 0.9, so at threshold 0.9 **every** prediction is
"below threshold" → **100% escalation** (not selective routing). Threshold 0.7 (≈10%
escalation, the genuinely uncertain cases) is the calibrated, meaningful setting.

| threshold | escalated (Ahmed) |
|---|---|
| 0.9 | 818 (100%) ← invalid |
| 0.8 | 481 (59%) |
| **0.7** | **84 (10%)** ← used |
| 0.6 | 41 (5%) |

## Methodological note
**Router thresholds must be calibrated per primary model, because probability scales
differ across models.** Our XLM-R primaries are over-confident (softmax peaks near
1.0), so threshold 0.9 escalates only ~20–28%; Ahmed's model peaks at ~0.86, so the
same 0.9 escalates 100%. A fixed escalation threshold does **not** transfer across
primaries — it must be set relative to each model's confidence distribution
(equivalently, calibrate confidences before routing).

## Interpretation
Even with calibrated escalation, the agentic layer is **slightly negative** on
Ahmed's strong primary (−0.0049 acc; net −4 on the escalated 84). This is **expected
and consistent with our cross-experiment finding that agentic gains depend on primary
strength**: weak generated primary (~0.70) → agents add +0.06; strong EESA XLM-R
(0.82) → +0.027; near-perfect topic (0.99) → ~−0.0003; **Ahmed (0.9254) → −0.005**.
At ~0.92 accuracy the primary is already near the ceiling of what the LLM agents can
contribute on these hard code-switched cases, so the agents (a minority of which
disagree correctly) net slightly hurt. **Recommendation: use Ahmed primary_only**; the
agentic layer is not worth it at this primary strength.

(Frozen primary only; no retraining; Ahmed's predictions unmodified.)
