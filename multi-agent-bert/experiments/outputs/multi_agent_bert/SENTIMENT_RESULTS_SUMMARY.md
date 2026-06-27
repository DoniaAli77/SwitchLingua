# Sentiment Results Summary (EESA)

Consolidated sentiment results on the **real EESA test set (818)**. No new
training/generation/agentic runs. Topic classification is tracked separately
(`EXPERIMENT_T1/T2…`) and is **not** mixed into this sentiment summary.
Date: 2026-06-27.

---

## Main comparison table — EESA test (818)
| System | Accuracy | Macro F1 | Weighted F1 | Type |
|---|---|---|---|---|
| **Ahmed model (external baseline)** | **0.9254** | **0.9207** | **0.9254** | external; provided predictions |
| Ahmed frozen-primary + full_agentic @0.7 | 0.9205 | 0.9153 | 0.9202 | external primary + our LLM agents |
| EESA-only XLM-R, Adafactor (E0) | 0.8533 | 0.8409 | 0.8530 | our best trained primary |
| EESA full_agentic reference (XLM-R + LLM agents) | 0.8509 | 0.8401 | — | our pipeline |
| EESA-only XLM-R, AdamW (original ref) | 0.8240 | 0.8088 | — | reference |
| Generated-only C3-960 full_agentic (seed 456) | 0.7543 | 0.7387 | 0.7515 | generated + LLM agents |
| Generated-only C3-960 primary (3-seed mean) | 0.6695 | 0.6592 | — | generated only |

Ranking: **Ahmed (0.9254) > our EESA models (≈0.85) > generated-only+agents (0.75) >
generated-only primary (0.67).**

---

## 1. Real EESA supervised results
Fresh `xlm-roberta-base` fine-tuned on real EESA train (2,463–2,464), evaluated on
EESA test.
| Model | Accuracy | Macro F1 |
|---|---|---|
| XLM-R, AdamW (original reference) | 0.8240 | 0.8088 |
| **XLM-R, Adafactor (E0, our best)** | **0.8533** | **0.8409** |
| XLM-R + LLM agents, full_agentic (threshold 0.9) | 0.8509 | 0.8401 |

- Switching optimizer **AdamW → Adafactor** added **+0.029 acc** (0.8240 → 0.8533).
- The multi-agent pipeline added a small **+0.027 acc** over the (AdamW) primary in
  the reference run; on a strong primary the agent headroom is modest.

## 2. Generated-only SwitchLingua results
Fresh XLM-R fine-tuned on **generated data only** (no real EESA), evaluated on EESA
test. Matched recipe (Adafactor).
| Train size | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| C1 — 240 generated | 0.5905 | 0.5619 | single seed |
| C2 — 480 generated | 0.6500 | 0.6345 | 3-seed mean |
| C3 — 960 generated | 0.6695 | 0.6592 | 3-seed mean |
| C3-960 best-dev + **full_agentic** | 0.7543 | 0.7387 | agents rescue +0.06 |

- **Generated-only training scales positively: 240 → 480 → 960** (≈0.59 → 0.65 → 0.67).
  (An earlier "480 > 960" reading was a single-seed artifact, retracted by the 3-seed
  check.)
- **The multi-agent pipeline strongly improves the weak generated-only model**: on the
  selected 960 checkpoint, full_agentic lifted **0.696 → 0.754 (+0.058)** — the
  largest agent rescue of any experiment (weak primary ⇒ most agent headroom).

## 3. Augmentation investigation (generated data added to real EESA)
Does generic generated data help when *mixed into* real EESA? Full detail in
`EXPERIMENT_E_AUGMENTATION_CONSOLIDATED_SUMMARY.md`.
- **Full data (matched, Adafactor):** EESA-only (E0) 0.8533 vs EESA+960 (E3) 0.8411 →
  augmentation **did not help** (−0.012; E3's apparent gain over the AdamW ref was the
  optimizer, not the data).
- **Low-resource (fixed +960):** hurt most when generated **dominated** the mix
  (10% real: −0.034 at 80% generated; ≈0 once generated is a minority).
- **Ratio sweep (20–50% generated):** all changes **within ±0.02 single-seed noise**;
  no ratio reliably helps; strong harm only when generated dominates.
- **Diagnosis — domain mismatch:** generated data is more balanced Arabic-English
  (CMI ~41 vs ~24), cleaner/more formal (MSA) vs EESA's dialectal noisy social-media,
  and shares only **~10%** of the EESA-test vocabulary. → generated data adds
  off-domain signal, not target-domain signal.
- **Conclusion:** generic generated data is **neutral as minority augmentation,
  harmful when dominant, and not reliably helpful** — its value is as **standalone**
  training data, not as an augmenter of a different-domain real set.

## 4. Ahmed external baseline
- **Accuracy 0.9254 · Macro F1 0.9207 · Weighted F1 0.9254** (per-class F1: positive
  0.953, negative 0.914, neutral 0.895). Full detail in
  `EXPERIMENT_AHMED_MODEL_BASELINE.md`.
- **Ahmed's result is an external baseline evaluated from provided predictions.** We
  did **not** reproduce Ahmed's full preprocessing/training pipeline locally.
- **After receiving Ahmed's exact test CSV, predictions were text-aligned and labels
  matched `y_true.npy` for all 818 samples.** Aligned file:
  `data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv`.
- **Ahmed's model is the strongest EESA sentiment baseline** (+0.072 acc / +0.080
  macro F1 over our best XLM-R primary). It is an **external** model, not part of our
  trained pipeline — but it can be plugged in as a **frozen primary** (below).

### 4b. Ahmed frozen-primary + full_agentic (our pipeline on top of Ahmed)
Ahmed's aligned predictions used as a **frozen primary**; the agentic layer evaluated
on top. Full detail in `EXPERIMENT_AHMED_FROZEN_PRIMARY_FULL_AGENTIC.md`.
| | accuracy | macro F1 |
|---|---|---|
| Ahmed primary_only | 0.9254 | 0.9207 |
| Ahmed full_agentic @ threshold 0.7 | 0.9205 | 0.9153 |
| **Δ (agentic)** | **−0.0049** | **−0.0054** |

- Escalated: **84 / 818 (10.3%)** · wrong→correct **11** · correct→wrong **15** ·
  **net −4** · cost ~$0.043.
- **The agentic layer slightly hurts Ahmed's strong primary** (the agents break a few
  more correct predictions than they fix on the hard escalated cases).
- **Threshold-calibration note:** threshold 0.9 (our XLM-R default) is **invalid for
  Ahmed** — his confidence maximum is **0.864**, so 0.9 escalates **100% (818/818)**,
  which is not selective routing. **Threshold 0.7** (≈10% escalation) was used for the
  fair selective-routing test. → **Router thresholds must be calibrated per primary
  model, because probability scales differ across models** (XLM-R peaks near 1.0;
  Ahmed peaks ~0.86); a fixed threshold does not transfer.

## 5. Main conclusions
1. **Best real-EESA sentiment model = Ahmed's external model (0.9254)** — a strong
   reference, not produced by our pipeline.
2. **Our best trained model = EESA-only XLM-R Adafactor (E0, 0.8533)**; the
   multi-agent pipeline adds only a small lift on this strong primary.
3. **SwitchLingua generated data is genuinely useful as a *standalone* source** — it
   carries real sentiment signal, **scales 240→480→960**, and the **multi-agent
   pipeline rescues it most where it is weakest** (generated-only 0.696 → 0.754).
4. **Generic generated data does not improve a real EESA model via naive augmentation**
   — neutral as a minority, harmful when it dominates — because of a domain/register
   mismatch, not a generation defect.
5. **Agent value tracks primary strength — the "primary-strength curve":**
   | primary | accuracy | agentic Δ acc |
   |---|---|---|
   | weak generated (C3-960) | ~0.70 | **+0.06** (substantial rescue) |
   | real EESA XLM-R (E0/ref) | 0.82–0.85 | **+0.027** (moderate help) |
   | **Ahmed (very strong)** | **0.9254** | **−0.005** (slightly hurts) |
   | topic XLM-R (near-perfect) | 0.99 | ~−0.0003 (neutral/slightly harmful) |

   → The agentic layer helps most where the primary is weakest and **slightly hurts a
   very strong primary** (Ahmed) — at high accuracy the primary is near the ceiling and
   the LLM agents add noise rather than signal.
6. **Router thresholds must be calibrated per primary model.** Threshold 0.9 was
   appropriate for our over-confident XLM-R primaries but **invalid for Ahmed**
   (confidence max 0.864 → 100% escalation); threshold 0.7 gave the fair
   selective-routing test (~10% escalation). Probability scales differ across models,
   so a fixed threshold does not transfer.

*Sentiment only. Topic (ARENTC) results are reported separately and not combined here.*
