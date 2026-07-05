# Design G on the WEAK C3 Primary — Results (real run)

Design G (Lexical + Polarity + Contextual + non-voting IntentGate, semantic_v1 prompts)
on the **weak C3 generated primary** (`sz960_seed456` XLM-R checkpoint, real transformer on
GPU), `full_agentic`, **threshold 0.90**, w_primary=1.0, gpt-4o-mini temp 0. Same EESA test
split (`eesa_sentiment_test.jsonl`, 818). This is the first run of the best parallel design
on the regime where the primary is weak. Date: 2026-07-02.

## Headline — the agents clearly help here

| system | accuracy | macro F1 | escalated acc | net vs primary |
|---|---|---|---|---|
| primary_only (C3) | 0.6956 | 0.6830 | 125/231 = 0.5411 | — |
| original full_agentic (default trio) | 0.7543 | 0.7387 | — | — |
| **Design G** | **0.7604** | **0.7469** | **178/231 = 0.7706** | **+53** |

- **vs primary_only: +0.065 accuracy, +0.064 macro F1** (622/818 vs 569/818).
- **vs original full_agentic: +0.006 / +0.008** — the parallel improvements (Polarity
  replacing Logic, the IntentGate, semantic_v1) **transfer to the weak primary** and add a
  little on top of the default agentic pipeline.
- Escalation: 231/818 = 28.2% at threshold 0.90.
- Cost: 1155 calls, **$0.176**.

## Escalated-subset transitions (where the agents act)
- **W→C = 65, C→W = 12, net = +53.** On the 231 escalated samples the agents **rescued 65
  wrong primary predictions and broke only 12.**
- Escalated accuracy lifted **0.5411 → 0.7706** (+0.23 absolute on the subset).
- McNemar on the discordant pairs (b=65, c=12): χ² ≈ 36.5, **p ≪ 0.001 — highly
  significant.** This is a real effect, not noise.

## Interpretation — the mirror image of Ahmed, exactly as predicted
- On **Ahmed** (strong primary) the primary was already ~0.75 on its escalated subset, so
  the ~0.75 agent ceiling left **no headroom** → v1 net −1, v2 net −11.
- On **C3** (weak primary) the primary is only **0.54** on its escalated subset, so the
  same ~0.77 agent ceiling is **far above it** → **net +53**.
- Same agents, same ~0.75–0.77 ceiling, opposite outcome — driven entirely by the primary's
  strength on the escalated subset. This **confirms the root-cause equation** (predicted C3
  gain ≈ +0.059; observed +0.065 for G) and validates the whole "the lever is the weak
  primary, not the aggregation topology" thesis with a live, significant result.

## What this establishes
1. **The agentic layer's value is real and large where the primary is weak** — +6.5 points
   overall, +23 points on the escalated subset, p ≪ 0.001.
2. **Design G is the parallel ceiling on C3: 0.7604 / escalated 0.7706.** This is the anchor
   the sequential runs (v2 next, v1 optional) must beat to claim forward-pragmatics helps
   *where the agents matter*.
3. The strong-primary null results (Ahmed) and this weak-primary positive are fully
   consistent — the system behaves exactly as the ceiling model says it should.

## Caveats
- **Seed-456 is the best-dev C3 checkpoint** (primary above the 960-seed mean), so its
  primary_only 0.6956 is a touch high; the agentic *gain* would likely be even larger on a
  mean-seed primary. The **relative** G-vs-primary comparison here is on the same checkpoint,
  so it is clean.
- Single temp-0 draw (±1–2 sample noise); the +53 escalated net is far outside noise.
- `decided_by`/gate-intervention counts not serialized this run; a cheap re-capture (like
  the Ahmed attribution scripts) would break down how much came from the IntentGate vs the
  Polarity swap.

## Artifacts
- Metrics/predictions: `experiment_G_c3/G_c3__{primary_only,full_pipeline}_*`
- LLM usage: `experiment_G_c3/G_c3__llm_usage.json`
- Run log: `experiment_G_c3_run.log`
- Comparators: original C3 full_agentic `experiment_C3_generated_960/full_agentic_seed456/`,
  `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md` (the prediction this confirms).
