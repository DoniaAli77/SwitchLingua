# Results — slide-ready tables

Two tracks: (1) SwitchLingua thesis (frozen) · (2) Multi-Agent BERT Experiment C (active). Model = gpt-4o-mini for generation; xlm-roberta-base for the classifier.

================================================================================
# TRACK 1 — SwitchLingua (task-aware generation; per-sentence vs aggregate scoring)
================================================================================

## Slide A1 — Masking: per-sentence catches what aggregate hides
| Acceptance bar | Sample | Masking rate |
|---|---|---|
| 7.0 | 54 scenarios | 41.5% |
| 7.0 | 101 scenarios | **35.6%** |
| 8.0 (default) | 101 scenarios | 0% (scores packed ~7) |
**Takeaway:** at a calibrated bar (7.0), aggregate scoring lets a weak sentence escape in ~36–42% of scenarios.

## Slide A2 — Refiner effectiveness
| Test | Result | Stat |
|---|---|---|
| Within-sentence improvement | **+0.60** points, 79/87 (90.8%) | p ≈ 0 |
| Your refiner vs original refiner (head-to-head) | tie | p = 0.53 |
**Takeaway:** the contribution is *what gets refined* (detection/routing), not a better rewriter.

## Slide A3 — Task-aware generation quality (Test 1)
| Task | Task-correct | CS-valid | CS-ratio MAE vs 70% |
|---|---|---|---|
| topic | **100%** | 87–100% | 14–23 pts |
| sentiment | 70% | 87–100% | 14–23 pts |
| NER (English-only) | **40%** | 87–100% | 14–23 pts |
**Takeaway:** topic strong, sentiment moderate (neutral drag), NER weak (under-produces English-script PER); CS-ratio off target.

## Slide A4 — TaskValidator necessity (Test 2, real validator)
| Policy | Accepted | Task-wrong accepted | Precision |
|---|---|---|---|
| quality only | 86 | 25 | 70.9% |
| quality + TaskValidator | 62 | **9** | **85.5%** |
**Takeaway:** validator cuts task-wrong accepts 25→9; benefit concentrated in NER, ~null for sentiment (neutral evades both).

## Slide A5 — Quality scoring alone is task-blind
| Signal | When task is WRONG |
|---|---|
| fluency | ~8 / 10 |
| naturalness | ~8 / 10 |
**Takeaway:** high fluency/naturalness even on task-failing sentences → motivates the TaskValidator.

## Slide A6 — CS-ratio measurement (Test 4, partial; 30-sentence set)
| Method | Self-consistency | Variance |
|---|---|---|
| Deterministic counter (ours) | exact | **0** |
| LLM-only counting (×3 repeats) | **disagrees on 12/30 (40%)** | ~0.6 tok / 2.3% |
| Binary code-switch agreement (det vs LLM) | 0/30 mismatch | — |
**Takeaway:** the LLM counter is non-reproducible; the deterministic counter is exact. (Human-accuracy MAE pending.)

================================================================================
# TRACK 2 — Multi-Agent BERT, Experiment C (synthetic → real transfer)
================================================================================

## Slide C1 — Generated dataset (240, accepted)
| Property | Value |
|---|---|
| size / balance | 240 — **80 / 80 / 80** (pos/neg/neu) |
| duplicates | 0 |
| CS-valid (recomputed) | 240 / 240 |
| TaskValidator passed | 240 / 240 |
| quality score range | 7.0 – 8.4 |
| source mix | v3 run 141 · pilot_v1 94 · early run 5 |
| cs_ratio mix | 70%: 99 · 60%: 72 · 50%: 69 |

## Slide C2 — Training setup
| Item | Value |
|---|---|
| base model | xlm-roberta-base (fine-tuned, not from scratch) |
| train samples | 240 |
| epochs / optimizer | 4 / Adafactor |
| eff. batch / fp16 | 16 / on |
| final train loss | 0.89 |

## Slide C3 — Results on REAL EESA test (818) — headline
| Model (train data) | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Exp C — 240 synthetic** | **0.590** | **0.562** | ≈0.584 |
| Exp A — real EESA (2,464) xlm-roberta | 0.831 | 0.819 | 0.831 |
| Exp A — real EESA mBERT | 0.807 | 0.790 | 0.806 |
| majority baseline (all-positive) | 0.444 | — | — |
**Takeaway:** +15 pp over majority from 240 synthetic; 24-pp gap to real-data training = expected weak-transfer (10× less data, synthetic→real shift, optimizer diff). Not a bug.

## Slide C4 — Exp C per-class (EESA test)
| Class | Precision | Recall | F1 | support |
|---|---|---|---|---|
| positive | 0.66 | 0.75 | 0.70 | 363 |
| negative | 0.58 | 0.46 | 0.51 | 197 |
| neutral | 0.49 | 0.46 | 0.47 | 258 |

## Slide C5 — Exp C confusion matrix (rows = true, cols = predicted)
| true ↓ / pred → | positive | negative | neutral |
|---|---|---|---|
| **positive** | 274 | 22 | 67 |
| **negative** | 48 | 90 | 59 |
| **neutral** | 95 | 44 | 119 |
**Takeaway:** not collapsed (predicts all 3); errors = negative/neutral leaking to positive/neutral.

## Slide C6 — CS-validity diagnosis & config-only fix
| Config | cs_ratio / cs_type | CS-valid rate |
|---|---|---|
| v1 (baseline) | 70% / Intra+Inter | **30%** (99.6% of failures fully-Arabic) |
| v2 pilot | 50/60/70 / Intra-only | 43% |
| v3 at scale | 50/60 / Intra-only | 50% ≈ **49%** · 60% ≈ 40% |
**Takeaway:** 70% Arabic target was too heavy; lower ratio + intrasentential-only ≈ doubles usable yield. Filter never loosened.

## Slide C7 — Scaling 240 → 480 (160/label) — current
| Label | Pool (pre-balance) | Target 160 | Yield (kept/scenario) |
|---|---|---|---|
| positive | 181 | ✅ | 1.76–1.83 |
| neutral | 156 | +4 | 1.68–1.90 |
| negative | **147** | **+13** | 1.37–1.45 |
**Status:** 2 windows done; paused on daily API quota; one small window finishes it. (480 not built/trained yet.)

================================================================================
# One-glance summary
================================================================================
| Track | Result | State |
|---|---|---|
| SwitchLingua masking | 35.6% @ bar 7 | ✅ shown |
| SwitchLingua refiner | +0.60 (routing win, rewrite tie) | ✅ |
| SwitchLingua validator | precision 70.9→85.5% | ✅ (NER) |
| SwitchLingua CS-counter | LLM 40% self-disagree vs 0 variance | 🟡 human acc pending |
| Exp C transfer (240) | 0.590 acc / 0.562 macro-F1 (vs 0.831 real) | ✅ diagnosed = expected |
| Exp C data scaling | 240 → 480 (pool 181/156/147) | ⏳ in progress |
| Multi-Agent BERT training (480) | — | ❌ not yet |
