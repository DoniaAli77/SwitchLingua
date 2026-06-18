# Experiment D — Ahmed's EESA models as the primary classifier (PLAN ONLY)

**Goal:** swap XLM-R for each of Ahmed's three pre-trained EESA models
(M7_HBA, M17_HTH, M25_Ensemble) as the pipeline **primary**, run the full
multi-agent pipeline on EESA test, and see whether results beat the XLM-R numbers.
Three **separate** experiments. Date: 2026-06-18. **Nothing executed yet.**

---

## 1. What these models actually are (investigated)
- **Framework:** TensorFlow / **Keras 2.15** (`w.h5`, `weights-saved.hdf5`) — *not*
  PyTorch/HuggingFace. The pipeline is PyTorch.
- **Architecture** (`params.json`): char-CNN + char-BiLSTM ⊕ **AraBERT-Twitter word
  features (1324-dim)** → BiLSTM(256) → **custom `SelfAttention`(200)** → Dense →
  softmax over **3 classes** (positive/negative/neutral). Sentence-level sentiment. ✔ matches our label set.
- **Two engineered inputs**, not text:
  - `input_1`: (116, 1324) precomputed AraBERT-Twitter + emotion/NE features per token.
  - `char_input`: (116, 15) char indices (299-char vocab, 15 chars/word).
- **Ahmed's code is present:** `ahmed models/sentimentAnalysisTransFusionGCV2.ipynb`
  — contains the `SelfAttention` layer, preprocessing (`removeLinksHashTags`,
  `replace_consecutive_emojis`, `pureArabic/pureEnglish`, …), `generateTestData`,
  feature build, `load_model`, `model.predict`. **This is the linchpin.**
- **Only error rows shipped:** `testErrorAnalysis.csv` = *misclassified* samples
  only (70 / 69 / 61 rows), **no probabilities, no full predictions.** Can't drive
  the pipeline directly → we must **re-run inference**.

## 2. Two blockers found
- **B1 — No TF installed.** `tensorflow` / `keras` are absent from `.venv`. Need
  TF + Keras 2.15 (and matching `numpy`); AraBERT-Twitter must be downloadable
  (HF: `aubmindlab/bert-base-arabertv02-twitter`) — corporate SSL proxy may bite.
- **B2 — Test-set mismatch.** Pipeline evaluates **818** samples
  (`eesa_sentiment_test.jsonl`); Ahmed's "~90%" was on the **~1099-row** raw
  `EESA-Test.csv` (processed csv = 1092). **To compare with XLM-R (0.8240 / 0.8509)
  we must re-run Ahmed's models on the same 818 subset** — their headline 90% is on
  a different (larger) set and is not directly comparable.

## 3. Recommended design — offline precompute, then a thin adapter
Keep TensorFlow **out** of the pipeline. Do inference once, offline, and feed the
results in.

**Phase 1 — offline TF inference (per model), reusing Ahmed's notebook code**
For each of the 818 EESA-test sentences: preprocess → build `input_1` (AraBERT-Twitter
features) + `char_input` → `model.predict` → softmax. Save per-sample:
`{sample_id, text, pred_label, probabilities{pos,neg,neu}, confidence}` to
`experiment_D/<model>/primary_predictions.jsonl`. One file per model.
*(confidence = max softmax prob — this is what the router needs.)*

**Phase 2 — new adapter `PrecomputedPrimaryClassifier`** (small, ~40 lines)
Implements the same interface as `PrimaryTransformerClassifier`
(`run(state) -> state`, sets `ModelOutput(label, confidence, probabilities)`), but
looks the values up from the Phase-1 file by `sample_id` (or normalized text).
No TF, no Keras inside the pipeline.

**Phase 3 — run the pipeline (unchanged) per model**
`primary_only` then `full_agentic` (Fix #2 on, signal off — our locked defaults),
exactly like the XLM-R runs, but with `--primary_model precomputed
--precomputed_predictions <file>`. Agents + consensus are untouched.

Why this design: isolates the messy TF/feature work to one reproducible offline
step, leaves the pipeline (and its 897 tests) essentially untouched, and is
identical across all three models.

## 4. Experiment matrix
| Model | primary_only | full_agentic | notes |
|---|---|---|---|
| M7_HBA | ✔ | ✔ | char-CNN+BiLSTM+attn |
| M17_HTH | ✔ | ✔ | variant |
| M25_Ensemble | ✔ | ✔ | fewest dev/test errors → likely strongest |

`primary_only` = free (no API). `full_agentic` = API per escalated sample; cost
depends on each model's confidence calibration (see Risk R3).

## 5. Comparison target (same 818 EESA test, kept separate from XLM-R tables)
| System | acc | macro F1 |
|---|---|---|
| XLM-R primary_only (ref) | 0.8240 | 0.8088 |
| XLM-R full_agentic best | 0.8509 | 0.8401 |
| M7_HBA / M17_HTH / M25 primary_only | ? | ? |
| …full_agentic | ? | ? |

Headline question: does a stronger/different primary lift the *full pipeline*, and
do the agents still add value on top of a ~90% primary?

## 6. Risks / decisions to resolve BEFORE running
- **D1 (gating): how do we get probabilities on the 818 set?**
  (a) **Ask Ahmed** to run his notebook on the 818 subset and hand us a CSV of
  `text,pred,prob_pos,prob_neg,prob_neu` → skips ALL TF work (best case), or
  (b) **we reproduce** Phase 1 ourselves from the notebook (needs TF install +
  AraBERT-Twitter download + the exact char vocab + feature artifacts).
- **D2 — char vocab + feature artifacts.** The 299-char map and any cached
  embedding/feature `.npy` are **not in the repo**; confirm they're in the notebook
  or obtainable, else Phase 1 can't reproduce the training-time features exactly
  (mismatch would degrade accuracy).
- **D3 — env.** Install TF+Keras 2.15 in a *separate* venv (avoid disturbing the
  PyTorch pipeline env); AraBERT download may hit the SSL proxy.
- **R3 — confidence calibration.** These models' softmax confidence ≠ XLM-R's;
  the escalation threshold (0.8/0.9) may need re-tuning per model, or ~95%
  escalation like Exp C if poorly calibrated. Plan a quick confidence histogram
  before committing to a threshold.
- **B2 — test alignment.** Re-run on the 818 jsonl (recommended) so results are
  comparable to XLM-R; report Ahmed's own ~1099 number only as context.

## 7. Effort / sequence
1. **Resolve D1** (Ahmed's probs vs reproduce) — decides 80% of the effort.
2. If reproduce: stand up TF venv, port notebook → a script `infer_ahmed_models.py`,
   verify it reproduces Ahmed's reported numbers on his own test split (sanity), then
   run on the 818 subset.
3. Build `PrecomputedPrimaryClassifier` + wire a `--primary_model precomputed` flag.
4. Run primary_only ×3 (free) → check accuracy vs 90% claim + confidence histograms.
5. Pick threshold(s) → run full_agentic ×3 (paid) → compile Experiment D report
   (separate from A/C and the XLM-R ablation tables).

---
**Open question for you (D1):** can Ahmed produce per-sample **probabilities** for
the 818-sample EESA test (or share the precomputed test feature `.npy` + char
vocab)? That single answer decides whether Phase 1 is a 1-hour adapter job or a
multi-step TF-reproduction effort. Nothing will run until you confirm the path.
