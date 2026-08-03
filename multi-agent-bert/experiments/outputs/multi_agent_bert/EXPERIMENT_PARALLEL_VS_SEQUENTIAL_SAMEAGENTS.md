# Fair Parallel-vs-Sequential — Same Agents (Ahmed, Design C)

Isolates the effect of **inter-agent ordering alone**. Prior sequential runs
(V1/V2) changed the agents, the prompts, AND the fusion, so they could not
attribute the gap to topology. Here the exact three parallel specialists (Lexical
+ Polarity + Contextual), their prompts, the Ahmed precomputed primary, threshold
0.70, gpt-4o-mini, and the confidence-weighted consensus vote are ALL held fixed;
the single change is `task_config.sequential_chain` (False → True). When True the
three agents run in a fixed order and each later agent receives an agent-only
chain block of the earlier specialists' conclusions (no primary signal). Date:
2026-07-19.

## Result

| metric | Parallel (independent) | Sequential (chained) | Δ |
|---|---|---|---|
| accuracy (818) | **0.9254** | 0.9242 | −0.0012 |
| macro F1 (818) | **0.9211** | 0.9198 | −0.0013 |
| escalated accuracy (84) | **0.7500** (63/84) | 0.7381 (62/84) | −1 sample |
| predictions changed | — | 7 / 818 | — |

**Sequential ordering, with identical agents, did not help — it trended slightly
negative (net −1 correct sample).** The magnitude (7 changed, net −1) is within
the ±1–2-sample temperature-0 run-to-run noise, so the honest reading is **no
improvement**, not a significant regression.

## Why (mechanism) — the 7 changed samples

| sample | true | parallel | sequential | outcome |
|---|---|---|---|---|
| 00100 | negative | neutral | negative | HELPED |
| 00193 | negative | neutral | negative | HELPED |
| 00542 | positive | neutral | positive | HELPED |
| 00239 | neutral | neutral | positive | HURT |
| 00310 | neutral | neutral | positive | HURT |
| 00396 | neutral | neutral | negative | HURT |
| 00449 | neutral | neutral | negative | HURT |

helped = 3, hurt = 4. Every HURT case is `true=neutral` that the sequential chain
pulled into a polar label; every HELPED case is a true-polar that the chain
correctly pulled off neutral. This is the **correlated-agents / anchoring**
effect: once a later agent sees an earlier agent's polar read, it is pulled
toward it — which helps when the earlier agent is right and hurts when it is
wrong. On this data the parallel design's independence (which better preserves
neutral) was marginally more valuable than the chain's shared context.

## Interpretation for the thesis
This is the controlled test the V1/V2 comparison could not provide. It shows the
earlier finding — parallel voting ≥ sequential — is **not merely an artifact of
V1/V2 using different agents/prompts/fusion**: even with the *same* agents and the
*same* consensus vote, sequential wiring is at best neutral and trends slightly
negative. Independence between voters is a feature of the parallel design, not an
incidental detail. Report as: "holding agents, prompts, primary, and consensus
fixed, sequential inter-agent ordering produced no accuracy or macro-F1 gain
(−1 sample, within run-to-run noise)."

## Replication at the best config — G2 selective gate, gpt-4.1-mini

Same test on the strongest Ahmed configuration (Lexical + Polarity + Contextual
voters + non-voting selective IntentGate, gpt-4.1-mini). `sequential_chain` chains
the three voters; the gate stays independent and post-consensus.

| metric | Parallel | Sequential | Δ |
|---|---|---|---|
| accuracy (818) | **0.9315** | **0.9315** | 0 |
| macro F1 (818) | **0.9277** | **0.9277** | 0 |
| escalated accuracy (84) | **0.8095** (68/84) | **0.8095** (68/84) | 0 |
| predictions changed | — | 2 / 818 | — |

**Identical accuracy and macro-F1.** Two predictions changed and exactly cancelled
(1 helped: 00182 neutral→negative✓; 1 hurt: 00073 negative→neutral✗), net 0.
So at a stronger model + gate, sequential ordering makes **literally no
difference** — reinforcing the Design-C finding (net −1). Across both configs,
chaining identical agents is **neutral at best**.

## Replication at Design G (full gate), gpt-4o-mini

| metric | Parallel | Sequential | Δ |
|---|---|---|---|
| accuracy (818) | 0.9267 | 0.9279 | +0.0012 |
| macro F1 (818) | 0.9227 | 0.9242 | +0.0015 |
| escalated accuracy (84) | 0.7619 (64/84) | 0.7738 (65/84) | +1 sample |
| predictions changed | — | 3 / 818 | — |

Here sequential is +1 sample (2 helped, 1 hurt). Note the canonical Design-G run
(0.9279 / 0.9242, 65/84) matches this SEQUENTIAL number exactly, and this parallel
run drew 64/84 — i.e. the direction is a one-sample artifact, not a real effect.

## Cross-config summary — the direction is noise

| Config | Parallel F1 | Sequential F1 | Net (esc) |
|---|---|---|---|
| Design C, 4o-mini | 0.9211 | 0.9198 | seq −1 |
| Design G, 4o-mini | 0.9227 | 0.9242 | seq +1 |
| G2 gate, 4.1-mini | 0.9277 | 0.9277 | 0 |

The sign of the difference **flips across configs** (−1, +1, 0), averaging ≈ 0.
With gpt-4.1-mini's own temp-0 non-determinism independently measured at ±1 sample
(re-running the same G2 config gave 761 vs 762/818), the parallel-vs-sequential
gap is **indistinguishable from run-to-run noise in every config**. Conclusion:
holding agents + fusion fixed, **sequential inter-agent ordering has no measurable
effect** on this task. Some per-sample effects are consistent (e.g. 00100 always
helped, 00449 always hurt by chaining) but they cancel in aggregate.

## Sequential agent-ORDER sweep (Design C, gpt-4o-mini)

Does the *order* in which the chained agents see each other matter? Holding
everything fixed (`sequential_chain=True`, same agents/prompts/primary/consensus)
and varying only `agent_stage_order`:

| order | accuracy | macro F1 | escalated | fallbacks |
|---|---|---|---|---|
| L→P→C (default) | 0.9230 | 0.9183 | 61/84 | 0 |
| C→P→L (reverse) | 0.9242 | 0.9204 | 62/84 | 0 |
| P→L→C | 0.9230 | 0.9186 | 61/84 | 0 |
| C→L→P | 0.9230 | 0.9188 | 61/84 | 0 |

**macro-F1 spread across all four orders = 0.0021** (0.9183 → 0.9204), i.e. a
single escalated sample (61 vs 62/84). No ordering is systematically better; the
best (reverse) leads by exactly one sample, within the ±1-sample temp-0 noise.
0 fallbacks — no network contamination. Conclusion: **the order of the chained
agents does not matter** for this task.

## Chain FRAMING test — does an anti-anchoring instruction suppress collaboration?

The sequential chain block ended with anti-anchoring wording ("do your own
analysis FIRST … do NOT simply copy them"), which could bias the chained agents
toward independence and manufacture the null result. Tested by adding a
`sequential_chain_style` knob and re-running Design C three ways in one session:

| mode | acc | macro F1 | escalated | changed vs parallel |
|---|---|---|---|---|
| parallel | 0.9254 | 0.9211 | 63/84 | baseline |
| sequential-independent (anti-anchoring) | 0.9242 | 0.9195 | 62/84 | 5 |
| sequential-collaborative ("build on your teammates") | 0.9230 | 0.9183 | 61/84 | 6 |

**The framing is not inert** — the collaborative version changed *more* predictions
(6 vs 5), so the agents genuinely engaged with the prior conclusions more. But the
extra collaboration made it **slightly worse, not better**: parallel 63 >
independent 62 > collaborative 61 (monotonic). The changed samples explain why —
both framings correctly fix the same 2 true-polar cases (00100, 00542), but the
collaborative framing produces *more* neutral→polar errors (4 hurt vs 3): when the
agents are told to build on each other's polar reads, they **amplify each other's
over-calling** on genuinely neutral text. This is the correlated-agents failure
mode, now shown with an explicit collaboration knob.

**Conclusion:** the null result is robust to framing. Encouraging collaboration
does not recover a benefit — it slightly amplifies shared errors, reinforcing that
parallel voting's independence is the safer design. (All within ±2 samples, so
"no benefit / trends negative," not a proven regression.)

## Story-aligned framing test on DESIGN G (the endpoint config)

Design C isolates the chain cleanly, but the thesis narrative reaches **Design G**,
so the framing test was repeated there (full gate, gpt-4o-mini). Parallel and
sequential-independent reused from the Design-G parallel-vs-sequential run;
collaborative is new.

| mode | acc | macro F1 | escalated | vs parallel |
|---|---|---|---|---|
| parallel | 0.9267 | 0.9227 | 64/84 | baseline |
| sequential-independent | 0.9279 | 0.9242 | 65/84 | +1 (3 changed) |
| sequential-collaborative | 0.9267 | 0.9227 | 64/84 | 0 (2 changed, 1↑1↓) |

On Design G the collaborative run **exactly ties parallel** (64/84). Combined grid:

| | parallel | seq-independent | seq-collaborative |
|---|---|---|---|
| Design C | 63/84 | 62/84 | 61/84 |
| Design G | 64/84 | 65/84 | 64/84 |

Differences are ≤1 sample and flip sign between configs → run-to-run noise. No
mode consistently wins, at either config, under either framing.

## COMPLETE GRID — all designs × all modes (Ahmed, 818 / 84 escalated)

| Design | Model | Mode | Accuracy | Macro F1 | Escalated /84 |
|---|---|---|---|---|---|
| C | 4o-mini | parallel | 0.9254 | 0.9211 | 63 |
| C | 4o-mini | sequential-independent | 0.9242 | 0.9195 | 62 |
| C | 4o-mini | sequential-collaborative | 0.9230 | 0.9183 | 61 |
| G | 4o-mini | parallel | 0.9267 | 0.9227 | 64 |
| G | 4o-mini | sequential-independent | 0.9279 | 0.9242 | 65 |
| G | 4o-mini | sequential-collaborative | 0.9267 | 0.9227 | 64 |
| G2 | 4.1-mini | parallel | 0.9315 | 0.9277 | 68 |
| G2 | 4.1-mini | sequential-independent | 0.9315 | 0.9277 | 68 |
| G2 | 4.1-mini | sequential-collaborative | 0.9303 | 0.9262 | 67 |

Escalated-only grid:

| Design | parallel | seq-independent | seq-collaborative |
|---|---|---|---|
| C | 63 | 62 | 61 |
| G | 64 | 65 | 64 |
| G2 | 68 | 68 | 67 |

Findings: (1) seq-independent ≈ parallel, differences flip sign across designs
(−1/+1/0) → noise. (2) seq-collaborative is **never better** than parallel (−2 / 0
/ −1) → collaboration does not help and trends slightly negative. (3) Sample
**ahmed-eesa-00449** (a neutral sentence) was over-called negative by the chain in
*every* sequential run across all three designs — the consistent signature of
correlated over-calling. Parallel voting is retained.

## Reproduce
- Code: `task_config.sequential_chain` (default False); block in
  `src/prompts/_agent_chain_block.py`; injected in `llm_lexical_agent.py`,
  `polarity_agent.py`, `contextual_agent.py`. Tests:
  `tests/test_sequential_chain_block.py` (7 tests).
- Run: `scripts/ahmed_parallel_vs_sequential_sameagents.py`.
- Artifacts: `experiment_ahmed_parallel_vs_sequential_sameagents/{records.json,parallel_vs_sequential_report.txt}`.

## Caveats
- Single run per mode, temperature 0; the ±1–2-sample noise exceeds the observed
  −1 gap, so treat as "no improvement," not a proven regression.
- Design C (no IntentGate) — the cleanest 3-voter ablation. Adding the gate (G)
  is a separate axis and was not varied here.
- The fresh Parallel-C (0.9211 F1) differs from the earlier Design-C ablation
  (0.9226 F1) by ~1 sample, consistent with temp-0 non-determinism.
