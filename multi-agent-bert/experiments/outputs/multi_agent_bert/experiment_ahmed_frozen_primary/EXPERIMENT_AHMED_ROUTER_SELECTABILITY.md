# Could a Smarter Router Have Selected Only the Beneficial Escalations? (Ahmed)

The 84 Ahmed-escalated samples treated as a dataset: each labeled
beneficial/harmful/neutral, then tested whether **router-visible (primary-side)**
features distinguish the beneficial cases from the harmful ones. Analysis only — no
LLM calls, no training, no generation. Date: 2026-06-27.

## Labeling (84 escalated)
A router decides **before** agents run, and beneficial/harmful is defined by the
correctness *change* escalation caused:
| outcome | definition | count |
|---|---|---|
| **beneficial** | wrong→correct (agents fixed Ahmed) | **11** |
| **harmful** | correct→wrong (agents broke Ahmed) | **15** |
| neutral | correct→correct or wrong→wrong (no correctness change) | 58 |

Equivalently: beneficial ⇔ **Ahmed was wrong** (and agents right); harmful ⇔ **Ahmed
was right** (and agents wrong). Among the 84, Ahmed is **correct 63 / wrong 21**. So a
"smart router" would need to escalate the 21 Ahmed-wrong cases and skip the 63
Ahmed-right cases — i.e. **predict Ahmed's own errors from primary features alone.**

## Do primary features separate Ahmed-correct from Ahmed-wrong? — NO
Mean primary-side features (all available to the router at decision time):
| group | confidence | top-2 margin | entropy | text len |
|---|---|---|---|---|
| Ahmed CORRECT (63) | 0.588 | 0.338 | 0.935 | 11.1 |
| Ahmed WRONG (21) | 0.589 | 0.333 | 0.932 | 12.2 |

**The distributions are essentially identical** — confidence 0.588 vs 0.589, margin
0.338 vs 0.333, entropy 0.935 vs 0.932. The primary gives the router **no signal** about
where it is wrong on this subset.

Threshold sweeps confirm it (overall Ahmed-wrong base rate = 25%):
| rule | n | Ahmed-wrong % | beneficial | harmful |
|---|---|---|---|---|
| margin < 0.10 (most "uncertain") | 9 | 33% | — | — |
| margin < 0.20 | 18 | 28% | — | — |
| margin < 0.30 | 31 | 26% | — | — |
| conf < 0.50 | 17 | 29% | 2 | 5 |
| conf < 0.60 | 41 | 22% | 5 | 10 |
| conf < 0.65 | 55 | 24% | 8 | 11 |

Tightening the confidence/margin cutoff does **not** concentrate Ahmed's errors (stays
~25%), and at **every** cutoff **harmful ≥ beneficial** — there is **no confidence or
margin threshold at which escalation becomes net-positive.**

## The only weak signal — predicted class — still loses
| Ahmed predicts | n | Ahmed acc | beneficial | harmful |
|---|---|---|---|---|
| positive | 15 | 0.80 | 2 | 2 |
| negative | 37 | 0.78 | 2 | 2 |
| **neutral** | 32 | **0.69** | 7 | **11** |

Ahmed is slightly less accurate when predicting **neutral** (0.69 vs ~0.79), and that is
where nearly all the action is — but **escalating Ahmed-neutral is still net-negative**
(11 harmful vs 7 beneficial). A class-based router rule (escalate only neutral) would
reduce volume but **not** flip the sign.

## Oracle vs. achievable router
- **Oracle router** (escalate only the 11 truly-beneficial): net +11 → full-test
  0.9254 → **0.9388**. This is the unreachable upper bound (requires knowing the outcome).
- **Best achievable router** from primary features: because beneficial and harmful are
  indistinguishable, any feature-based subset keeps **harmful ≥ beneficial** → the best
  it can do is **escalate nothing → net 0 → recover Ahmed's 0.9254.** It cannot reach
  net-positive.

## Conclusion
**A smarter router could NOT have selected the beneficial subset.** The features a
router can see (Ahmed's confidence, probability margin, entropy, predicted class, text
length) are **statistically identical between the cases Ahmed gets right and the cases
it gets wrong** on this escalated subset, so the 11 beneficial escalations are
**indistinguishable** from the 15 harmful ones at routing time.

**Root cause:** Ahmed's softmax is poorly calibrated / compressed on these hard
code-switched cases — within the narrow 0.4–0.7 confidence band where everything
escalated, confidence does **not** track correctness (Ahmed is ~75% right regardless of
whether his confidence is 0.5 or 0.65). The router's only input (confidence) carries no
information about Ahmed's actual error locations.

**Implication for the framework:** for a strong primary, the agentic layer cannot be
*rescued by a smarter router* — the best a router can do is escalate nothing. This
strengthens the project conclusion: **a smarter router helps only when the primary's
confidence is informative about its own errors;** for a strong, poorly-calibrated
primary like Ahmed, the right policy is simply **don't escalate** (or strictly
non-overriding consensus). To *enable* selective routing one would first need to
**calibrate the primary's confidence** so that low confidence actually predicts error —
that, not the consensus rule, is the prerequisite for a useful router here.

## Artifacts
- Derived from `error_attribution/attribution_table.json` (per-sample Ahmed + agent
  outputs); feature computation in the analysis above.
