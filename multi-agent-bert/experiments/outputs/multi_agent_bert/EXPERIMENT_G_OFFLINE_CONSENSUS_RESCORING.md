# Offline Consensus-Rescoring — Can Any Restricted Aggregation Rule Beat G?

Pure **offline simulation** over the saved Design-G capture (per-agent labels + gate outputs
already stored). **No paid LLM calls, no retraining, no generation.** Rules are **fixed by label
patterns** (they never consult the true label to decide whether to fire); predictions are scored
against truth only after the rule is applied. Date: 2026-07-01.

## Method & honest caveats
- **Data:** `experiment_ahmed_designG_intent_gate/error_attribution/attribution_table.json` (84
  escalated: lexical/polarity/contextual/gate labels + gated final) + primary predictions for all
  818. Full accuracy = 734 non-escalated (primary) + 84 escalated (rule-applied).
- **Baseline = this G capture draw**, which landed **66/84 escalated (net +3, full 0.9291)** — one
  sample above the headline G (65/84, net +2, 0.9279) due to gpt-4o-mini temp-0 noise. All rules
  are compared **on the same table**, so the *relative* comparison is exact; "beats G" means beats
  this baseline (and therefore headline G too).
- **w_primary sweep is a LABEL-PROXY:** per-agent confidences were **not serialized** in the G
  capture, so the confidence-weighted consensus cannot be reproduced. The proxy (agents = 1 vote,
  primary = w votes, primary tie-break) does **not** reproduce G at w=1.0 (proxy 0.9242 ≠ G 0.9291),
  so treat its **absolute** numbers as unreliable; only its **qualitative** trend is trustworthy.

## Results
| # | rule | full acc | macro F1 | wtd F1 | esc | W→C | C→W | net | interv | helped | hurt | vs G |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **baseline (G)** | 0.9291 | 0.9257 | 0.9292 | 66/84 | 10 | 7 | **+3** | — | — | — | = |
| 2 | Contextual neutral guard | 0.9291 | 0.9257 | 0.9292 | 66/84 | 10 | 7 | **+3** | **0** | 0 | 0 | **= (never fires)** |
| 3 | Lexical strong-cue protection | 0.9279 | 0.9242 | 0.9279 | 65/84 | 9 | 7 | +2 | 1 | 0 | **1** | **worse (−1)** |
| 4 | Combined (2+3) | 0.9279 | 0.9242 | 0.9279 | 65/84 | 9 | 7 | +2 | 1 | 0 | 1 | **worse (−1)** |
| 5a | w_primary = 1.0 (proxy) | 0.9242 | 0.9197 | — | 62/84 | 9 | 10 | −1 | 6 | 1 | 5 | worse |
| 5b | w_primary = 1.5 (proxy) | 0.9242 | 0.9197 | — | 62/84 | 9 | 10 | −1 | 6 | 1 | 5 | worse |
| 5c | w_primary = 2.0 (proxy) | 0.9242 | 0.9197 | — | 62/84 | 9 | 10 | −1 | 6 | 1 | 5 | worse |
| 5d | w_primary = 3.0 (proxy) | 0.9254 | 0.9207 | — | 63/84 | 0 | 0 | 0 | 18 | 7 | 10 | = primary_only |

Confusion pairs changed: Rule 3/4 **broke** `neutral→negative` ×1 (fixed 0). Rule 5d **fixed** 7
and **broke** 10 (net 0 = recovers the primary exactly, 0 overrides).

## Reading each rule
- **Rule 2 — Contextual neutral guard: 0 interventions.** It never fired. Every case where
  Contextual = neutral, primary = neutral, and consensus went polar is **already caught by the
  IntentGate** (which protects a neutral primary from polar overrides). A second neutral guard on
  the strongest agent is therefore **fully redundant** — it recovers nothing the gate hasn't. This
  is a clean, informative negative: **the gate already saturates the neutral-protection domain.**
- **Rule 3 / 4 — Lexical strong-cue protection: fires once, hurts once (−1).** The single case where
  "primary + Lexical agree on a polar label but consensus neutralized it" was a case where the
  neutralization was **correct** (a false polar cue the panel rightly dropped). Restoring the polar
  broke it (`neutral→negative`). This is exactly the **low-precision polar dissent** measured earlier
  (Lexical lone-dissent precision 0.37): protecting Lexical's polar cue adds errors.
- **Rule 5 — w_primary sweep: no gain.** At w = 1–2 the (label-proxy) majority over-corrects → net
  −1; at w = 3 the primary dominates → consensus = primary_only (net 0, 0 overrides). This matches
  the confidence-based original simulation (`EXPERIMENT_AHMED_CONSENSUS_SIMULATION.md`): raising
  w_primary monotonically trades overrides for primary-recovery and **tops out at the primary**,
  never above it.

## Verdict — **no rule beats G. Honestly.**
- **Rule 2 ties G exactly** (0 interventions — redundant with the gate).
- **Rules 3/4 are worse** (−1; Lexical protection undoes a correct neutralization).
- **Rule 5 is worse or, at best, recovers primary_only** (net 0).

No fixed, label-only aggregation rule improves on G. This **confirms the design prediction**: the
consensus loss is real (oracle 0.83–0.88 on the escalated subset) but **not recoverable by
label-based re-fusion**, because (a) the one domain where recovery is safe — neutral protection —
is **already saturated by the IntentGate**, and (b) the other specialists' dissents are
low-precision (Lexical 0.37, Polarity 0.33), so protecting/trusting them **adds** errors. G is at
the achievable aggregation optimum on the strong primary given saved (label-only) information.

## What this rules out — and what remains
- **Ruled out (offline, free):** Contextual neutral guard (redundant), Lexical cue protection
  (harmful), naive w_primary re-tuning (no gain). No paid run of these is warranted.
- **Not assessable offline (needs new paid captures):** confidence-aware arbitration and a learned
  meta-selector both need per-agent **confidences**, which weren't serialized — and prior evidence
  says the agents' self-confidence is poorly calibrated, so even a re-capture is low-expected-value.
- **The remaining real lever is not aggregation on the strong primary.** Every strong-primary result
  is now at ceiling (root-cause + consensus-loss + this rescoring all agree). The decisive open
  experiment is **G (and v3) on the weak C3 generated primary**, where the consensus loss is larger
  and the primary term is small — the one regime where the agentic layer, and any aggregation
  refinement, can actually pay off.

## Artifacts
- Simulator: computed inline from `experiment_ahmed_designG_intent_gate/error_attribution/
  attribution_table.json` (labels only) + `ahmed_eesa_test_predictions_aligned.csv`.
- Basis: `EXPERIMENT_CONSENSUS_LOSS_AGGREGATION_DESIGN.md` (rule design + lone-dissent precision),
  `EXPERIMENT_CONSENSUS_LOSS_ANALYSIS.md`, `EXPERIMENT_AHMED_CONSENSUS_SIMULATION.md` (w_primary,
  confidence-based), `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md`.
