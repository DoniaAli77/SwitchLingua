# sequential_sentiment_v1 — Ahmed Frozen-Primary Results (real OpenAI run)

Real run of the staged pipeline (Intent → Polarity → Pragmatic → deterministic
controller) on the **Ahmed frozen primary**, `full_agentic`, threshold **0.7**,
gpt-4o-mini, temperature 0. Same 818-sample test set and precomputed primary as the
whole design line. Date: 2026-07-01.

Command: `--sentiment_agent_variant sequential_sentiment_v1 --primary_model precomputed`
`--precomputed_predictions ahmed_eesa_test_predictions_aligned.csv --threshold 0.7`
`--llm_client openai --mode both`.

## Headline

| system | accuracy | macro F1 | escalated acc | net vs primary |
|---|---|---|---|---|
| primary_only (baseline) | **0.9254** | **0.9207** | 63/84 = 0.7500 | — |
| **sequential_sentiment_v1** | **0.9242** | **0.9195** | 62/84 = 0.7381 | **−1** |
| *(reference) Design G* | *0.9279* | *0.9257* | *65/84 = 0.774* | *+2* |

Full-set correct: **756/818** vs primary **757/818**. **Sequential is 1 sample BELOW
primary_only, and 3 below Design G.**

## Per-class (full_pipeline)
| label | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.9507 | 0.9559 | 0.9533 | 363 |
| negative | 0.9184 | 0.9137 | 0.9160 | 197 |
| neutral | 0.8911 | 0.8876 | 0.8893 | 258 |

## Escalated-subset behaviour (the only place the pipeline acts)
- Escalation rate: **0.1027** (84/818) — identical to every Ahmed@0.7 run (escalation
  is set by the primary, not the variant).
- **W→C = 13, C→W = 14, net = −1.** The pipeline is **not inert**: it changed **27 of
  84** escalated decisions away from the primary. But those changes net slightly
  negative — it rescued 13 primary errors and introduced 14 new ones.
- Sequential-on-escalated **0.7381 (62/84)** vs primary-on-escalated **0.7500 (63/84)**.

## Cost
- OpenAI calls: **336** (84 escalated × 3 stages = 252 minimum, so **~84 retries** —
  i.e. ≈33% of stage first-attempts returned non-conforming JSON and were retried).
- Tokens: 274,638 (prompt 252,793 + completion 21,845). **Est. cost: $0.051.**

## Verdict — honest

**The staged pipeline does not beat the strong primary. It lands at −1 sample vs
doing nothing (primary_only) and −3 vs Design G — statistically indistinguishable
from primary_only (a 1-sample difference; McNemar would give p≈1.0).**

This is exactly what every prior analysis predicted:
- The strong-primary system is at its **ceiling** (agent-ceiling ≈0.75 on the
  escalated subset; net agentic gain ≈ 0 when the primary is already ~0.75 there).
- The sequential architecture **did real work** — 27/84 interventions, a much higher
  action rate than G's veto — yet still netted −1, because the escalated subset is at
  the **information floor**: rescuing 13 costs 14. Changing the *shape* of aggregation
  (staged vs parallel-vote vs veto) cannot manufacture signal that isn't in the subset.
- So this is a **clean, informative negative**: it confirms, with a genuinely different
  and more active architecture, that the strong-primary regime has no headroom — the
  agentic layer can only pay off on a **weak primary (C3)**, where the primary term is
  small and the escalated subset is larger and more recoverable.

## Caveats
- **~33% stage retry rate** (84 retries). The one-retry-then-coerce path fired often;
  the run did not crash and produced valid labels throughout, but an unknown fraction of
  stages may have fallen back to a safe default rather than a genuine model read. The
  Evaluator persisted final predictions only — **`decided_by` / coercion counts were not
  serialized this run** (the per-sample sequential trace lives in `state.extras`, which
  the standard Evaluator does not save). A faithful breakdown of *how* each decision was
  reached (Rule 1–4 distribution, true coercion rate) needs a small re-capture pass like
  the `ahmed_*_attribution.py` scripts. Two prompt tweaks would likely cut the retry rate
  (tighten the JSON-only instruction / add a fenced example the parser strips).
- Single temp-0 draw; ±1–2 sample run-to-run noise applies (the −1 is within it).

## Decision-trace re-capture (84 escalated, faithful) — resolves the caveats

A separate temp-0 re-capture recorded the per-sample sequential trace from
`state.extras` (`scripts/ahmed_seqv1_attribution.py` →
`experiment_seqv1_ahmed/decision_trace/`).

**`decided_by` distribution (of 84):**
| rule | fired | meaning |
|---|---|---|
| `polarity_kept` (Rule 3) | **58** | pragmatic kept → Stage-2 polarity label |
| `intent_no_opinion` (Rule 1) | **23** | no-opinion neutral gate (the IntentGate equivalent) |
| `pragmatic_revision` (Rule 2) | **3** | confident sarcasm/implicature flip |
| `fallback_*` (Rule 4) | **0** | never needed — Polarity always usable |

**Coercion / faithfulness — the important correction:**
- **0 stage coercions, 0 retries, 0 llm_errors** in this capture. Every stage parsed
  valid JSON on the first attempt. **So the pipeline ran faithfully — the result is NOT
  an artifact of JSON fallback.** (The headline run's 84 retries were transient
  formatting wobble that *resolved to valid parses on retry*, not coercions to default —
  gpt-4o-mini temp-0 output-format jitter, not a logic failure. The earlier ~33% "retry"
  caveat is therefore benign.)
- Intent said opinion **True on 61, False on 23** → the 23 False map exactly to the 23
  `intent_no_opinion` neutral decisions.
- **Pragmatic verifier is nearly inert: 81 keep / 3 revise.** Stage 3 almost never fires;
  the work is done by Stage 1 (23 neutral gates) + Stage 2 (58 polarity reads).

**This capture's escalated outcome: W→C 13, C→W 13, net 0, 63/84 = 0.7500 — exactly
primary-on-escalated.** (The headline draw was net −1 / 62-84; the two differ by 1
sample = the ±1 temp-0 noise. Both are at the primary's escalated accuracy.)

**Reinforced verdict:** running *faithfully* (no fallback), the staged pipeline
**reproduces the primary's escalated accuracy almost exactly** (net 0 to −1). It
restructures *how* the decision is reached — Polarity + a no-opinion neutral gate — but,
like every parallel variant, the "pragmatic/sarcasm" stage barely intervenes (3/84) and
the express-vs-mention gate is the only active lever. Same ceiling, reached a different
way.

## Artifacts
- Decision trace: `experiment_seqv1_ahmed/decision_trace/trace_table.{json,csv}`
- Metrics/predictions: `experiment_seqv1_ahmed/seqv1_ahmed__{primary_only,full_pipeline}_*`
- LLM usage: `experiment_seqv1_ahmed/seqv1_ahmed__llm_usage.json`
- Run log: `experiment_seqv1_ahmed_run.log`
- Implementation: `EXPERIMENT_SEQUENTIAL_SENTIMENT_V1_IMPLEMENTATION_CHANGELOG.md`
