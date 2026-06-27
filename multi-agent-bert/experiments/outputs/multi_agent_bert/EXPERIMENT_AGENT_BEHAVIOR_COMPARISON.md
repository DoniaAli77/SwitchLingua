# Agent Behavior Across Sentiment Experiments — When Does the Agentic Layer Help?

Analysis only — **no new LLM calls, no training, no generation.** Compares agent
behaviour across the three completed full_agentic sentiment experiments, using saved
predictions. Date: 2026-06-27.

> **Data availability.** Per-agent outputs (individual lexical/logic/contextual labels)
> were persisted **only for Ahmed** (recovered via the attribution re-run). The C3 and
> EESA full_agentic runs saved only the *final* prediction + the primary's
> probabilities. So **Section C (override behaviour) and primary-vs-final accuracy are
> available for all three**; **Sections A/B/D (agent diversity / per-agent accuracy /
> correlation) are available for Ahmed only.** Recovering A/B/D for C3/EESA would
> require an LLM re-run (out of scope here). Conclusions below distinguish *measured*
> from *inferred*.

## Section C — override behaviour on the escalated subset (ALL THREE, measured)
| experiment | escalated | esc % | **primary acc (esc)** | **final acc (esc)** | W→C | C→W | net | full-test Δ |
|---|---|---|---|---|---|---|---|---|
| **C3 generated-960** (th 0.9) | 231 | 28.2% | **0.5411** | 0.7489 | 71 | 23 | **+48** | +0.059 |
| **EESA XLM-R** (th 0.9) | 190 | 23.2% | **0.5579** | 0.6737 | 42 | 20 | **+22** | +0.027 |
| **Ahmed frozen** (th 0.7) | 84 | 10.3% | **0.7500** | 0.7024 | 11 | 15 | **−4** | −0.005 |

(net / 818 = full-test Δ exactly: +48→+0.059, +22→+0.027, −4→−0.005.)

**The decisive pattern:** the **final consensus accuracy on the hard escalated cases is
roughly constant (~0.67–0.75)** across all three experiments — it is set by the LLM
agents' own competence on hard code-switched sentiment, **not** by the primary. What
*varies* is the **primary's accuracy on those escalated cases**:
- C3 primary 0.54 → agents lift to 0.75 (**+0.21**) → big rescue.
- EESA primary 0.56 → agents lift to 0.67 (**+0.12**) → moderate help.
- Ahmed primary 0.75 → agents drop to 0.70 (**−0.05**) → slight harm.

→ There is an **"agent ceiling" ≈ 0.70** on these escalated CS cases. The agentic layer
helps when the primary is **below** that ceiling and hurts when it is **above** it.

## Sections A / B / D — Ahmed only (measured)
**A. Agent diversity (84 escalated):**
| all 3 agree | exactly 2 agree | all disagree |
|---|---|---|
| **77 (92%)** | 7 (8%) | 0 (0%) |

Pairwise agreement: **Lexical–Logic 96% · Lexical–Contextual 94% · Logic–Contextual 93%.**

**B. Per-agent accuracy on the escalated subset:**
| primary | lexical | logic | contextual | final consensus |
|---|---|---|---|---|
| **0.7500** | 0.7143 | **0.6786** | 0.7262 | 0.7024 |

**D. Correlation:** the three agents behave as **one highly-correlated bloc** (92–96%
pairwise), and **all three are individually weaker than the Ahmed primary** (0.68–0.73
vs 0.75), so the bloc's combined weight (3) outvotes the stronger primary → consensus
0.702 < primary 0.750.

**A/B/D for C3 and EESA: NOT AVAILABLE** (per-agent outputs not persisted). The agent
*ceiling* (final accuracy ~0.67–0.75) is the same order across all three, which is
consistent with the same correlated agents operating everywhere — but the per-agent
correlation in C3/EESA is **inferred, not measured.**

## Answering the questions
**1. Were the agents also highly correlated in C3 and EESA?**
**Not directly measurable** from saved data (per-agent labels weren't stored). Measured
only for Ahmed (92%). It is *likely* similar — they are the same GPT-4o-mini agents, and
the near-constant final-accuracy ceiling (~0.70) across all three is consistent with a
similar correlated bloc — but this is an inference, not a measurement.

**2. If the agents were (likely) correlated everywhere, why did they still help in C3/EESA?**
Because **correlation is not what decides help vs harm — relative accuracy is.** In
C3/EESA the agent bloc (~0.67–0.75 on escalated) is simply **more accurate than the weak
primary** (0.54–0.56), so even a correlated bloc that votes together is voting *more
often correctly* than the primary → it rescues it. Correlation only became a problem for
Ahmed because there the bloc is **less accurate than the primary**, so its correlated
*errors* systematically override a better answer.

**3. What changed for Ahmed?** The **primary**, not the agents. Ahmed's escalated-subset
accuracy (0.75) is **above the agent ceiling (~0.70)**; C3/EESA primaries (0.54–0.56) are
far below it. The agents did not get worse — the primary got better than them.

**4. Dominant factor — primary strength, agent diversity, or both?**
**Primary strength (relative to the agent ceiling) is dominant and sets the *direction*
(help vs harm).** Agent correlation is (for Ahmed, measured; elsewhere inferred) a
near-constant that sets the *magnitude/decisiveness*: a correlated bloc votes together,
amplifying the rescue when it's right (weak primary) and the damage when it's wrong
(strong primary). **Both matter, but only primary strength flips the sign.**

## E. Final conclusion — one evidence-based rule for the whole project
> **The multi-agent consensus delivers a roughly fixed accuracy (~0.70, the "agent
> ceiling") on hard escalated code-switched cases, set by the LLM agents' own
> competence. It therefore HELPS when the primary's accuracy on those cases is below
> ~0.70 (weak/generated primaries: C3 0.54 → +0.059, EESA 0.56 → +0.027), is NEUTRAL
> near the crossover, and HURTS when the primary is above ~0.70 (strong primary, Ahmed
> 0.75 → −0.005). Because the agents act as a highly-correlated bloc (92% agreement,
> measured for Ahmed), they vote decisively in whichever direction — amplifying both the
> rescue and the harm.**

**Practical rule:** route to the agentic layer only where the primary is genuinely
weaker than the agents (escalated-accuracy < ~0.70 ⇔ weak/uncertain primary). For a
strong primary (Ahmed, 0.9254 overall), the agentic layer is **unnecessary and slightly
harmful** — it should be off or strictly non-overriding (see the consensus-simulation
report: agent-bloc + high `w_primary` recovers the primary exactly). This single
mechanism explains every sentiment result: weak generated primary → big agent rescue;
real EESA primary → modest help; very strong Ahmed primary → slight harm.

## Caveat
- A/B/D (agent diversity/accuracy/correlation) are **measured only for Ahmed**; for
  C3/EESA the agent-ceiling inference rests on Section C (final-vs-primary accuracy) +
  the same-agents argument. A no-cost-now-but-LLM-needed re-run of the C3/EESA escalated
  samples would let us *measure* their agent correlation and confirm the inference.
- Thresholds differ (C3/EESA 0.9, Ahmed 0.7) because escalation must be calibrated per
  primary (Ahmed's confidence maxes at 0.864); this affects *how many* escalate, not the
  per-case help/harm mechanism above.

## Artifacts
- Section C computed from the three full_agentic prediction files.
- Ahmed A/B/D from `error_attribution/attribution_table.json`.
