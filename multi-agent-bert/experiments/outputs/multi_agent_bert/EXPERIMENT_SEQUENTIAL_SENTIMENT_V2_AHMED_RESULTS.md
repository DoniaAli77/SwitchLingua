# sequential_sentiment_v2 — Ahmed Frozen-Primary Results (real OpenAI run)

Real run of the **forward-pragmatics** pipeline (Intent → Pragmatic FEATURES →
feature-aware Polarity → controller) on the Ahmed frozen primary, `full_agentic`,
threshold 0.7, gpt-4o-mini, temp 0. Same 818-sample test set / precomputed primary as v1.
Date: 2026-07-01.

## Headline — v2 is substantially WORSE

| system | accuracy | macro F1 | escalated acc | net vs primary |
|---|---|---|---|---|
| primary_only | **0.9254** | **0.9207** | 63/84 = 0.7500 | — |
| sequential_sentiment_**v1** | 0.9242 | 0.9195 | 62/84 = 0.7381 | −1 |
| **sequential_sentiment_v2** | **0.9120** | **0.9076** | **52/84 = 0.6190** | **−11** |
| *Design G (ref)* | *0.9279* | *0.9257* | *65/84 = 0.774* | *+2* |

Full-set correct: **746/818** (v2) vs 757 (primary) vs 756 (v1). **v2 loses 11 samples vs
doing nothing.** Escalated accuracy collapsed to **0.619** — *below* even the ~0.75 agent
ceiling. Cost: 334 calls, **$0.045**.

## What went wrong — the cascade risk materialized
- v2 **changed 31/84** escalated decisions (more active than v1's ~27 and far more than the
  primary). **W→C = 8, C→W = 19.** It broke more than twice what it fixed.
- **10 of the 19 breakages are correct-neutral → wrong-polar** (neutral→positive 6,
  neutral→negative 4). v2 **over-calls polarity**: fed pragmatic features
  (`implicit_stance`, `sarcasm_or_irony`), the resolver reads a stance into
  mention/description/meta cases the primary correctly called neutral.
- 5 more are polar→neutral over-neutralizations; 4 are polarity inversions
  (neg→pos 2, pos→neg 2) — consistent with false `sarcasm_or_irony=true`.
- Label distribution on escalated shifted from primary {neg 37, neu 32, pos 15} to v2
  {neg 39, **neu 24**, pos 21} — the neutral mass leaked into both poles.

## Interpretation — v1 and v2 now bracket the finding

This is the important result, and it is **not** what the v2 design hoped for:

- **v1 (anchored review):** inert (3/84 revisions), net ≈ 0/−1 → *reproduces* the primary.
- **v2 (forward, un-anchored):** active (31/84 changes), net **−11** → *destroys* signal.

**Removing the confirmation anchor did exactly what §6 of the v2 design warned about, at
full force.** The anchoring in v1 was, accidentally, *protective* — it suppressed harmful
flips. v2 unleashed the pragmatic features to drive the decision, but on the strong primary
the escalated subset is at the **information floor** and gpt-4o-mini's pragmatic feature
extraction is **unreliable**, so more activity = more damage. The base model over-detects
implicit stance and sarcasm, and once those features *drive* (rather than *review*) the
label, they convert correct neutrals into wrong polars.

**Combined lesson across v1 + v2:** on the strong primary, the agentic layer cannot add
signal, and its *net effect scales with how much it intervenes* — inert (v1) ≈ neutral,
active (v2) = harmful. The only regime where intervention can pay is the **weak primary**,
where the escalated subset actually carries recoverable signal. The strong-primary ceiling
is now confirmed from **five** independent directions (root-cause equation, parallel
ablations, v1-faithful-reproduction, weight/rescoring sweeps, and now v2's active-harm).

## Caveats
- `decided_by` / feature-usage were **not serialized** by the standard Evaluator, so the
  exact split (how many flips came from `sarcasm_or_irony=true` vs `implicit_stance` vs the
  no-opinion gate) is inferred from the label transitions, not measured. A cheap
  decision-trace re-capture (like `ahmed_seqv1_attribution.py`, ~$0.05) would pin the
  mechanism precisely — **not run here.**
- Single temp-0 draw; ±1–2 sample noise applies, but a **−11** gap is far outside noise.
- The 334 calls imply retries occurred; whether any stage coerced is not visible without the
  trace capture (v1's clean re-capture showed 0 coercion, so this is likely benign format
  jitter again).

## Verdict
**v2 fails on the strong primary — worse than v1, worse than primary_only, by a wide,
non-noise margin (−11).** It is a clean, informative negative: it confirms that the
confirmation-bias fix works *mechanically* (Stage 3 became active) but that activity is
net-harmful here because the strong-primary escalated subset has no recoverable signal and
the model's pragmatic features are noisy. **This does not test the design's actual
hypothesis** — which was always that pragmatic features help on a **weak** primary. On the
question v2 was built to answer (does forward-pragmatics beat review-pragmatics *where the
agents matter*), the Ahmed run is uninformative by construction; only a **C3** run can
answer it. If anything, v2 is now the stronger argument for **not** running more
strong-primary experiments.

## Artifacts
- Metrics/predictions: `experiment_seqv2_ahmed/seqv2_ahmed__{primary_only,full_pipeline}_*`
- LLM usage: `experiment_seqv2_ahmed/seqv2_ahmed__llm_usage.json`
- Run log: `experiment_seqv2_ahmed_run.log`
- Design: `EXPERIMENT_SEQUENTIAL_SENTIMENT_V2_FORWARD_PRAGMATICS_DESIGN.md`
- Comparators: `EXPERIMENT_SEQUENTIAL_SENTIMENT_V1_AHMED_RESULTS.md`,
  `EXPERIMENT_STAGE3_PRAGMATIC_REDUNDANCY_ANALYSIS.md`
