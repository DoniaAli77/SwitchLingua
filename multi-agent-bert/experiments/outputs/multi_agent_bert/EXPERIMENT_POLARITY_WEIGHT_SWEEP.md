# Polarity-Weight Sweep — Can Upweighting a Voter Replace the Selective IntentGate?

Design H established that the gate's criteria are inert when placed inside a
voting agent at weight 1.0. The natural follow-up — and the obvious reviewer
question — is whether simply giving the Polarity agent **more voting power** lets
it win the meta/mention cases on its own, making the gate unnecessary.

Method: Design C (Lexical + Polarity@logic + Contextual, **no gate**) was run once
on the Ahmed frozen primary, capturing each agent's (label, confidence) and the
primary's softmax. Those captured outputs were then re-fused **offline** with the
production `ConsensusAgent` at a range of Polarity ("logic" slot) weights. One paid
pass therefore yields the entire curve, and differences *between weights are exact*
— the same agent outputs are reused, so no LLM variance is involved.
gpt-4.1-mini, threshold 0.70, semantic_v1, w_primary 1.0, 818 samples / 84
escalated, 0 fallbacks. Date: 2026-08-04.

## Results

| Polarity weight | Accuracy | Macro F1 | Escalated |
|---|---|---|---|
| primary_only | 0.9254 | 0.9207 | 63/84 |
| w = 1.00 (Design C) | 0.9267 | 0.9220 | 64/84 |
| w = 1.25 | 0.9254 | 0.9210 | 63/84 |
| w = 1.50 | 0.9267 | 0.9221 | 64/84 |
| w = 1.75 | 0.9267 | 0.9221 | 64/84 |
| w = 2.00 | 0.9267 | 0.9221 | 64/84 |
| **w = 2.50 (best)** | 0.9279 | 0.9236 | 65/84 |
| w = 3.00 | 0.9230 | 0.9184 | 61/84 |
| **Design G2 (gate), mean n=5** | **0.9306** | **0.9266** | **67.2/84** |
| G2 noise band (n=5 range) | 0.9291–0.9315 | 0.9251–0.9277 | 66–68/84 |

**No weight reaches the G2 band.** The best setting (w = 2.50) falls short on both
criteria: macro F1 0.9236 vs the 0.9251 floor, and 65/84 vs the 66 floor.

## Shape of the curve

| Weight range | Behaviour |
|---|---|
| 1.00 – 2.00 | Flat at 64/84 — additional weight changes essentially nothing |
| 2.50 | Small peak at 65/84 (+1 over Design C) |
| 3.00 | **Collapse to 61/84** — below Design C, and below primary_only on macro F1 |

At w = 1.50 the result is identical to w = 1.00 (64/84; ΔF1 = 0.0001): a 50 %
increase in Polarity's voting power changed nothing.

## Interpretation — symmetry vs asymmetry

| | Upweighting Polarity | Selective IntentGate |
|---|---|---|
| Effect on outcomes | **Symmetric** — Polarity prevails more often when correct *and* when incorrect | **Asymmetric** — can only withhold a move *away from* the primary |
| Best observed gain | +1 escalated sample (w = 2.50) | **+3 escalated samples** |
| Failure mode | Over-weighting degrades sharply (−3 at w = 3.00) | Bounded: can only err by restoring an incorrect primary (1 of 6 firings) |

Because a voter's weight applies uniformly to every decision it participates in,
raising it cannot selectively strengthen the cases where the agent is right. The
gate's benefit derives from acting *only* in a restricted circumstance and in a
single direction — a property no weight setting can reproduce.

## Consequence for the thesis
This closes the most likely reviewer objection to the gate ("why not just tune the
consensus weights?"). Together with Design H, two independent ablations now support
the same conclusion:

| Ablation | Manipulation | Result |
|---|---|---|
| Design H | Gate's criteria moved *into* a voting agent (w = 1.0) | No benefit (64/84, = no gate) |
| Weight sweep | Voting agent given *more power* (w up to 3.0) | Best 65/84, never reaches the gate's 67.2 |

The Selective IntentGate's contribution is therefore attributable to its
architectural position — a conditional, one-directional veto applied after the vote
— rather than to the content of its criteria or to the influence of any single voter.

## Reproduce
`scripts/ahmed_polarity_weight_sweep.py`; artifacts in
`experiment_ahmed_polarity_weight_sweep/{captured.json,polarity_weight_sweep_report.txt}`.
Weights are applied via `ConsensusAgent(weights={"primary": 1.0, "logic": w})`; the
Polarity agent occupies the `logic` slot in Designs B/C/G/G2.
