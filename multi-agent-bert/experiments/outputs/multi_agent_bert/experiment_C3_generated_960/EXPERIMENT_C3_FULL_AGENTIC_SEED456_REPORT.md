# Experiment C3 — full_agentic on selected 960 checkpoint (seed 456)

full_agentic on the real EESA test, using the best-validating 960 generated-data
checkpoint. Threshold 0.9, Fix-2 primary-aware consensus ON (w_primary=1.0),
agents_use_primary_signal=false, GPT-4o-mini. Date: 2026-06-23.

## 1. Selected seed + checkpoint (and why)
**Seed 456** — `experiments/checkpoints/seed_stability/sz960_seed456/`.
Selection criterion = **best dev macro F1** among the three 960 seeds (chosen on
held-out dev, not test):
| seed | dev macro F1 |
|---|---|
| 42 | 0.6398 |
| 123 | 0.6646 |
| **456** | **0.6820 ✅ best** |

(Note: seed 456 is the best-*validating* checkpoint, so its primary is above the
960 mean — keep that in mind when comparing to the C2/C3 *means* below.)

## 2. Primary_only (seed 456)
accuracy **0.6956** · macro F1 **0.6830** · weighted F1 0.6971.

## 3–7. Full_agentic (seed 456) — EESA test (818)
Clean run (0 connection / 0 quota errors).

| | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| primary_only | 0.6956 | 0.6830 | 0.6971 |
| **full_agentic** | **0.7543** | **0.7387** | **0.7515** |
| **Δ** | **+0.0587** | **+0.0557** | **+0.0544** |

Per-class (full_agentic):
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.837 | 0.832 | 0.834 | 363 |
| negative | 0.646 | 0.843 | 0.731 | 197 |
| neutral | 0.745 | 0.578 | 0.651 | 258 |

Confusion (rows = true, cols = predicted; pos / neg / neu):
```
            pred_pos  pred_neg  pred_neu
true_pos       302       33        28
true_neg         8      166        23
true_neu        51       58       149
```
Prediction distribution: positive 361 · negative 257 · neutral 200
(true 363 / 197 / 258).

## 8–12. Escalation / agent effect
- Escalated: **231 / 818 (28.2%)** at threshold 0.9
- Escalated-only accuracy (full_agentic): **0.749**
- **wrong→correct: 71** · **correct→wrong: 23** · **net +48**

The agents fixed 71 and broke 23 of the 231 escalated samples → **net +48** → exactly
the +0.0587 overall accuracy gain. Every class improved (pos F1 0.780→0.834,
neg 0.661→0.731, neu 0.609→0.651 vs this checkpoint's primary).

## 13. Cost
**~$0.12** (231 escalated × 4 calls ≈ 924 LLM calls, gpt-4o-mini).

## 14. Comparison (EESA test)
| system | accuracy | macro F1 | note |
|---|---|---|---|
| C1 generated-240 primary_only | 0.5905 | 0.5619 | single seed |
| C2 generated-480 primary_only | 0.6500 ± 0.016 | 0.6345 ± 0.017 | 3-seed mean |
| C3 generated-960 primary_only | 0.6695 ± 0.024 | 0.6592 ± 0.021 | 3-seed mean |
| C3-960 seed-456 primary_only | 0.6956 | 0.6830 | selected (best-dev) |
| **C3-960 seed-456 full_agentic** | **0.7543** | **0.7387** | **this run** |
| Exp A real-EESA XLM-R primary_only | 0.8240 | 0.8088 | reference |
| Exp A real-EESA full_agentic best | 0.8509 | 0.8401 | reference |

## Findings
- **The multi-agent pipeline gives a large rescue on the weak generated-data
  primary: +5.9 acc / +5.6 macro F1** (0.696 → 0.754), all classes improved.
- This is the strong-rescue end of the cross-experiment **primary-strength curve**:
  weak primary (this, 0.70) → **agents add ~+6 pts**; strong primary (EESA, 0.82) →
  ~+2.7 pts; near-perfect primary (topic, 0.99) → agents are noise.
- full_agentic on a 960-generated model (0.754) closes much of the gap to the
  **real-EESA primary** (0.824) — i.e., agents partly compensate for training on
  generated rather than real data, at ~$0.12.

Caveat: seed 456 is the best-dev checkpoint (primary above the 960 mean), so 0.754
is a favorable-case full_agentic number for 960, not a 3-seed mean.

Stopped after this run (per instruction). No Ahmed models; seed-sweep outputs not
overwritten (full_agentic written to `experiment_C3_generated_960/full_agentic_seed456/`).
