# Threshold Sweep Analysis — Experiment A

**Analysis only. No retraining. No router/architecture changes.** Run date:
2026-06-06.

## Question
Do the specialist agents hurt because the escalation threshold is too
**conservative** (too few cases reach the agents), or because the **agents are
weaker than the primary** model?

## Method
- Reused the existing fine-tuned checkpoints (`eesa_mbert`,
  `eesa_xlm_roberta_base`) and the existing EESA test set — no retraining.
- Ran `evaluate_pipeline.py` in `paper_style` on EESA test with the router
  threshold overridden via `--threshold` ∈ {0.6, 0.7, 0.8, 0.9}, for both
  checkpoints (8 runs total). Routing rule is unchanged: escalate iff
  `primary.confidence < threshold` ([router.py:33](../../../src/pipeline/router.py#L33)).
- "Primary prediction" per sample taken from the existing `primary_only` runs;
  "agent/consensus prediction" taken from each sweep run's escalated samples.
  Joined by `sample_id`.
- Outputs per run under `threshold_sweep/<model>/th_<t>/`.

Primary-only overall test accuracy (baseline, no escalation): **mBERT 0.7971**,
**XLM-R 0.8240**.

---

## Main sweep

`finAcc` = final pipeline accuracy · `primAcc_esc` = primary accuracy on the
escalated subset · `agtAcc_esc` = agent/consensus accuracy on the same escalated
subset · `C→W` = originally-correct primary predictions changed to wrong ·
`W→C` = originally-wrong primary predictions corrected · `net` = W→C − C→W.

### mBERT
| th | esc % | finAcc | macroF1 | weightedF1 | primAcc_esc | agtAcc_esc | C→W | W→C | net |
|---|---|---|---|---|---|---|---|---|---|
| 0.6 | 5.75 | 0.7958 | 0.7788 | 0.7935 | 0.4255 | 0.4043 | 12 | 11 | **−1** |
| 0.7 | 9.17 | 0.7812 | 0.7606 | 0.7769 | 0.4800 | 0.3067 | 27 | 14 | **−13** |
| 0.8 | 14.67 | 0.7665 | 0.7441 | 0.7606 | 0.4833 | 0.2750 | 43 | 18 | **−25** |
| 0.9 | 22.49 | 0.7384 | 0.7101 | 0.7282 | 0.5326 | 0.2717 | 75 | 27 | **−48** |

### XLM-R
| th | esc % | finAcc | macroF1 | weightedF1 | primAcc_esc | agtAcc_esc | C→W | W→C | net |
|---|---|---|---|---|---|---|---|---|---|
| 0.6 | 5.01 | 0.8142 | 0.7983 | 0.8117 | 0.4146 | 0.2195 | 13 | 5 | **−8** |
| 0.7 | 7.46 | 0.8105 | 0.7939 | 0.8070 | 0.4426 | 0.2623 | 21 | 10 | **−11** |
| 0.8 | 13.33 | 0.7848 | 0.7671 | 0.7795 | 0.4954 | 0.2018 | 45 | 13 | **−32** |
| 0.9 | 23.23 | 0.7408 | 0.7163 | 0.7304 | 0.5579 | 0.2000 | 89 | 21 | **−68** |

**Every metric degrades monotonically as the threshold rises**, for both models.
The more we escalate, the worse it gets.

---

## Key observations

1. **The primary beats the agents on the escalated subset at every threshold.**
   `primAcc_esc` (0.41–0.56) is consistently and substantially higher than
   `agtAcc_esc` (0.20–0.40). Even on the *low-confidence* cases the primary was
   unsure about, the agents are still much worse than just trusting the primary.

2. **Corrections are rare; regressions dominate.** `W→C` (agents fix a primary
   error) is always far smaller than `C→W` (agents break a correct primary
   answer). Net is negative at every threshold and grows steeply more negative.

3. **Agents collapse to the majority class (`positive`).** Per-class breakdown of
   the escalated subset (true class → where agents send it):

   - mBERT @0.9 — escalated **negatives** (n=70): agents predict positive **59**,
     neutral 11, negative **0** → C→W = 39, **W→C = 0**.
   - XLM-R @0.9 — escalated **negatives** (n=70): agents predict positive **59**,
     neutral 10, negative 1 → C→W = 46, **W→C = 0**.
   - Escalated **neutrals** are likewise pushed to positive (mBERT @0.9: 56/63 →
     positive). Escalated **positives** are the only class that benefits, because
     the agents' default *is* positive.

   The agents/consensus essentially default to `positive` rather than
   discriminating — `W→C` for the **negative** class is **0 in every cell**.

---

## Per-class effect (negative vs neutral)

The damage is concentrated exactly where the primary was already weakest:

- **Negative** is destroyed: agents never recover a negative (W→C = 0
  everywhere) and convert many correct negatives to positive. This is the single
  largest contributor to the accuracy / macro-F1 drop.
- **Neutral** also degrades (pushed to positive), though less severely than
  negative.
- **Positive** improves slightly — but only as an artifact of the agents'
  positive bias, not genuine skill.

Because negative + neutral losses outweigh positive gains, **macro F1 falls
faster than accuracy** (e.g. XLM-R 0.7983 → 0.7163 across the sweep).

---

## Verdict

> If higher thresholds improve performance → conservative routing was the issue.
> If higher thresholds hurt more → weak specialist agents/consensus.

**Higher thresholds hurt more — strongly and monotonically — for both models.**
Therefore the problem is **weak specialist agents / consensus, not conservative
routing.** The conservative 0.6 threshold is actually *minimizing* the harm by
keeping the agents away from all but ~5% of cases; it is not the cause of the
degradation.

Mechanism: the current non-trained agents (keyword lexical + regex logic +
TF-IDF/`label_echo` mock contextual → consensus) do not discriminate on these
code-switched comments and collapse to the majority `positive` class. Trusting
the fine-tuned transformer primary is better than escalating, even on its
lowest-confidence predictions.

### Implications (not acted on here)
- Lowering the threshold further or changing the routing signal (margin/entropy)
  will **not** help — the agents are the bottleneck.
- To make agents useful they would need to be **stronger than the primary on the
  escalated slice** (e.g. trained agents, a real LLM contextual agent, or
  class-balanced consensus that does not default to positive), or be restricted
  to a confidence band where they demonstrably beat the primary (none found
  here).
- `full_agentic` behaved essentially identically to `paper_style` in Experiment
  A, so adding the mock-LLM agents does not change this conclusion.

No code, router, or model was modified. Experiment C not run.
