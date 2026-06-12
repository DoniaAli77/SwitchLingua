# Threshold Ablation Plan — extend the 2×2 to threshold 0.9

**Plan only — no OpenAI calls, do NOT run until approved.** Tests whether the
0.8 conclusions (Fix #2 neutral on strong agents; Fix #3 induces anchoring
without benefit) hold at a higher escalation threshold. Date: 2026-06-12.

## Setup (unchanged from the 0.8 ablation)
XLM-R, full_agentic, EESA test, gpt-4o-mini. Same 2×2, same seam flags:
| Cell | `--consensus_primary_weight` | `--agents_use_primary_signal` |
|---|---|---|
| A original | 0 | (off) |
| B Fix #2 only | 1.0 | (off) |
| C Fix #3 only | 0 | on |
| D Fix #2+#3 | 1.0 | on |

Done: **0.8**. Proposed next: **0.9 only**.

## 1. Estimated calls / cost for the 0.9 2×2
XLM-R escalates **~23.2% at 0.9** (≈ **190 / 818** samples) vs ~13.3% (109) at 0.8.
full_agentic makes 4 LLM calls per escalated sample.

| Cell | escalated | calls (≈190×4) | cost (≈) |
|---|---|---|---|
| A (signal off) | ~190 | ~760 | ~$0.10 |
| B (signal off) | ~190 | ~760 | ~$0.10 |
| C (signal on)  | ~190 | ~760 | ~$0.11 |
| D (signal on)  | ~190 | ~760 | ~$0.11 |
| **Total** | — | **~3,040** | **~$0.42** |

(signal-ON cells cost ~10% more from the extra prompt block; rates from the 0.8
run: ~$0.000131/call off, ~$0.000147/call on.) Runtime ≈ ~25 min/cell ≈ **~100
min** total on a stable connection.

## 2. Why 0.9 is the most informative next threshold
- **It maximizes the slice the fixes act on.** At 0.8 only ~13% escalate, so Fix
  #2/#3 touch ~109 samples and their effects sat inside GPT run-to-run noise. At
  0.9, ~23% escalate (~190) — **nearly double** — so any real directional effect
  is far more likely to exceed noise. If the fixes truly do nothing, 0.9 confirms
  it on a bigger sample; if they do something, 0.9 is where it shows.
- **It stress-tests the design's hardest case.** At 0.9 we escalate
  *confident-but-below-threshold* primaries (conf up to 0.9). Fix #2 is
  confidence-scaled, so it anchors those strongly — this is exactly the "high
  primary confidence but still escalated" case from the proposal. It also gives
  the agents more high-confidence primaries to (potentially) copy → the **strongest
  test of the Fix #3 anchoring effect**.
- **It brackets the realistic operating range.** 0.8 + 0.9 cover the high-recall
  end where escalation actually matters. **Lower thresholds escalate fewer
  samples** (0.7 ≈ 7.5%, 0.6 ≈ 5%) → even smaller fix effects → least informative,
  which is why they're the fallback, not the next step.

## 3. Metrics to report (per cell)
Same as the 0.8 table **plus** two escalated-subset measures (more central now
that the escalated set is larger):
1. accuracy, 2. macro F1, 3. weighted F1, 4. per-class F1 (esp. negative/neutral),
5. escalation count/rate, 6. OpenAI calls + cost, 7. connection errors,
8. parse errors, 9. **anchoring proxy** (escalated final == primary_only
agreement), 10. **escalated-subset accuracy**, 11. **net W→C − C→W vs primary on
the escalated subset** (does the fix change how often agents correctly vs wrongly
override the primary?).

## 4. How to compare 0.8 vs 0.9
- **Ordering & spread:** does A ≥ B ≥ C ≥ D persist, and does the A→D spread
  *grow* with more escalation (a growing spread = a real effect emerging from
  noise; a flat tiny spread = the fixes are genuinely neutral here)?
- **Anchoring delta:** at 0.8 the signal raised anchoring +3–4 pts (62%→65–66%).
  Does that delta **hold, grow, or shrink** at 0.9? A larger delta with no accuracy
  gain strengthens "Fix #3 = copying, not skill".
- **Fix #2 threshold-dependence (the key hypothesis):** Fix #2 was neutral at 0.8.
  At 0.9 the primary anchors on ~80% more samples; if Fix #2 (B,D) now **cuts
  regressions** (higher escalated-subset accuracy / less negative net loss) vs
  A,C, that refines the story to "Fix #2 neutral at low escalation, protective at
  high escalation." If it's still flat, Fix #2 is simply neutral for strong agents.
- Apply the same **noise band** (~±0.002–0.003 acc); only call a difference real
  if it exceeds it on the larger 190-sample escalated set.

## 5. Stopping rule
- Run **0.9 only**.
- **STOP (no 0.6/0.7) if 0.9 confirms 0.8:** all cells within noise, Fix #3 still
  anchors-without-benefit, Fix #2 still neutral, recommendation unchanged (Fix #2
  default on / Fix #3 default off). Conclusion then holds across the operating
  range — 0.6/0.7 (even fewer escalations) would add no information.
- **Only then run 0.6/0.7 if 0.9 CHANGES the conclusion**, defined concretely as
  any of: a cell differs from A by **>0.005** acc *or* macro F1 (beyond noise);
  the A≥B≥C≥D ordering **flips**; the anchoring delta **changes sign**; or the
  default recommendation (Fix #2 on / Fix #3 off) would **flip**. In that case
  0.6/0.7 map the threshold dependence / find the crossover.

## Exact commands (when approved)
Identical to the 0.8 ablation with `--threshold 0.9` and
`--output_dir .../ablation_2x2_th0.9/{A,B,C,D}`. Connectivity-probe per cell; mark
and re-run any connection-contaminated cell. No code/default changes; seam flags
only. **Awaiting approval before any OpenAI call.**
