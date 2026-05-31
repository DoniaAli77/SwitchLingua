# Threshold Calibration Report — Masking / Per-Sentence Validation

**Input (pre-refinement, refiner OFF):** `C:\Users\Eng.Donia\Documents\matser\SwitchLingua\experiments\outputs\switchlingua\per_sentence\validation_raw\Arabic.jsonl`  
**Scenarios:** 101 usable of 104 (excluded 3 with <2 sentences) · **Sentences:** 458

## Why a sweep, and why the default bar (8.0) is too strict for gpt-4o-mini

gpt-4o-mini produces a **compressed score distribution** (sentences cluster around ~7). Under the pipeline default acceptance bar of **8.0**, almost no scenario is accepted, so masking cannot be observed — not because masking is absent, but because the bar sits above the model's score range. We therefore report the **full threshold sweep** rather than a single bar.

- At **8.0**: 0 scenarios accepted, masking 0.0% — the bar is effectively inoperative for this model.

## Calibrated operating point

We treat **7.0** as a *calibrated operating point* (set to the model's median sentence quality), **not** an arbitrary replacement for 8.0. It is reported alongside the whole curve so no single threshold is cherry-picked.

- At **7.0**: accepted 64/101 scenarios; **masking 35.6%** (95% CI 27.0–45.4%); weak-sentence leakage 16.4% of accepted sentences; monolingual leakage 2.1%.

## Full threshold sweep

| Bar | Accepted scen | Masked scen | Masking % (CI) | Weak-sent leak % | Monoling leak % | Valid CS % |
|----:|----:|----:|:----|----:|----:|----:|
| 6.5 | 100 | 24 | 23.8% (16.5–32.9) | 6.6 | 2.9 | 97.1 |
| 7.0 | 64 | 36 | 35.6% (27.0–45.4) | 16.4 | 2.1 | 97.9 |
| 7.5 | 8 | 8 | 7.9% (4.1–14.9) | 32.1 | 0.0 | 100.0 |
| 8.0 | 0 | 0 | 0.0% (-0.0–3.7) | 0.0 | None | None |
| 8.5 | 0 | 0 | 0.0% (-0.0–3.7) | 0.0 | None | None |

## Main contribution

The contribution is **detection**: per-sentence scoring surfaces weak sentence-level outputs that aggregate scenario-level scoring hides. At the calibrated bar, a non-trivial fraction of scenarios the aggregate rule would accept actually contain a sub-threshold sentence; the per-sentence rule catches exactly these. The full sweep is reported to avoid cherry-picking, and the masking signal is threshold-sensitive because the model's scores are tightly packed.

## Method notes

- Pre-refinement (refiner OFF) sentence scores only; post-refinement scores are never used here.
- Aggregate = mean of per-sentence weighted scores (same formula as the pipeline).
- A scenario is *masked* if aggregate >= bar but >=1 sentence < bar.
- Monolingual leakage / valid-CS use the deterministic `compute_true_cs_stats` CS detector.
- Thresholds are read from the config YAML; none are hardcoded.
