# G2-Lazy — Deferring the Selective IntentGate to the Post-Consensus Stage

In Design G2 the selective IntentGate is constructed as a fourth *pre-consensus*
specialist with vote weight 0.0: its LLM call is made on **every** escalated
sample, but its judgement is consulted only afterwards, by the post-consensus
guard. Because the gate's prompt receives only the raw text and the label set —
never the consensus label and never the primary's prediction — the call can be
**deferred** until the guard actually needs it, i.e. only when consensus has
overridden the primary. G2-lazy implements exactly that; the decision rule is
unchanged. Ahmed precomputed primary, gpt-4.1-mini, threshold 0.70, semantic_v1,
w_primary 1.0. Date: 2026-08-04.

## Results

| Config | Accuracy | Macro F1 | Escalated | Gate LLM calls |
|---|---|---|---|---|
| G2 (eager) mean, n=5 | 0.9306 | 0.9266 | 67.2/84 | 84 (always) |
| G2 noise band (n=5 range) | 0.9291–0.9315 | 0.9251–0.9277 | 66–68/84 | 84 |
| **G2-lazy** | **0.9303** | **0.9267** | **67/84** | **29 (65% saved)** |

0 fallbacks. Accuracy falls **inside** the measured G2 noise band.

## Decision-equivalence

| Metric | G2 (eager) | G2-lazy |
|---|---|---|
| Overrides blocked by the gate | 6 | **6** |
| Escalated correct | 66–68 (mean 67.2) | 67 |
| Gate invocations | 84 | 29 |

Identical gate behaviour (6 blocked overrides) with 55 fewer LLM calls. Only
29/84 escalated samples produced a consensus override; on the remaining 55 the
gate's judgement could not have changed anything, so the call is skipped.

## Cost

| Measure | G2 | G2-lazy | Saved |
|---|---|---|---|
| Gate calls | 84 | 29 | 65% |
| Specialist calls (3 voters + gate) | 336 | 281 | 16% |

## Why this matters for the thesis
1. **The description becomes literally accurate.** With the eager wiring, "after
   the consensus is produced, the Selective IntentGate determines whether …"
   misplaces the LLM call (which happens before consensus). Under G2-lazy the gate
   genuinely is a post-consensus component, so the natural description is correct
   without qualification.
2. **The architecture matches the concept.** The gate is presented as a
   conditional veto; lazy invocation makes it *behave* conditionally rather than
   being computed unconditionally and used conditionally.
3. **It is strictly cheaper** at equal accuracy, which is a defensible efficiency
   claim rather than a performance one.

Recommended wording:
> The gate is consulted only when the consensus label differs from the primary
> prediction; in the reported configuration this occurred for 29 of 84 escalated
> instances, so the gate's language-model call was required for 35 % of escalated
> inputs. Deferring the call in this way leaves the decision rule unchanged and
> yielded accuracy within the measured run-to-run variation of the eager variant
> (0.9303 vs 0.9306 mean, band 0.9291–0.9315).

## Reproduce
- Variant: `lexical_polarity_contextual_lazy_gate` (opt-in; G/G2 unchanged).
- Implementation: `IntentGateAgent(lazy_agent=...)` in
  `src/agents/intent_gate_agent.py` — invokes the gate agent inside the guard,
  after the override check. Wiring in `evaluate_pipeline.py` sets
  `polarity_agent = None` so the gate is not a pre-consensus stage.
- Tests: `tests/test_agent_design_variants.py` — 5 lazy-gate tests, incl. zero
  invocations when consensus agrees with the primary.
- Script: `scripts/ahmed_g2_lazy_gate.py`; artifacts in
  `experiment_ahmed_g2_lazy_gate/{records.json,g2_lazy_report.txt}`.
