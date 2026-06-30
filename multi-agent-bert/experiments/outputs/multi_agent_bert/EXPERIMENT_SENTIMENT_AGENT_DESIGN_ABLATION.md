# Sentiment Agent-Design Ablation (A / B / C / D) — Ahmed Frozen-Primary

Empirical comparison of four specialist decompositions on the **same** Ahmed
frozen-primary full_agentic setup (semantic_v1 prompts, threshold 0.7, Fix-2
consensus w_primary 1.0, gpt-4o-mini, no training, no generation). Only the agent
set changes; router and primary are untouched; consensus is unchanged except the
default-off `polarity` slot (weight 1.0 only for D). Date: 2026-07-01.

| design | agents |
|---|---|
| **A** | Lexical + Logic + Contextual (current semantic_v1 default-trio variant) |
| **B** | Polarity + Contextual (Lexical abstains) |
| **C** | Lexical + Polarity + Contextual (Logic → Polarity) |
| **D** | Lexical + Logic + Contextual + Polarity (4 votes) |

Baseline: **Ahmed primary_only = 0.9254 / 0.9207 macro F1** (escalated-subset
accuracy 0.7500).

---

## Headline comparison (full test, 818; escalated = 84 for all)
| metric | A (Lex+Logic+Ctx) | B (Pol+Ctx) | **C (Lex+Pol+Ctx)** | D (4-agent) |
|---|---|---|---|---|
| **accuracy** | 0.9230 | 0.9254 | **0.9267** | 0.9254 |
| **macro F1** | 0.9183 | 0.9212 | **0.9226** | 0.9211 |
| **weighted F1** | 0.9202 | 0.9254 | **0.9266** | 0.9254 |
| escalated-only acc | 0.726 | 0.750 | **0.762** | 0.750 |
| wrong→correct | 12 | 10 | 12 | 11 |
| correct→wrong | 14 | 10 | 11 | 11 |
| **net** | **−2** | **0** | **+1** | **0** |
| neutral→negative breaks | 5 | **4** | 4 | 4 |
| neutral→positive breaks | 4 | **1** | 2 | 2 |
| total breaks | 14 | **10** | 11 | 11 |
| cost / calls | $0.0498 / 336 | **$0.0348 / 252** | $0.0494 / 336 | $0.0635 / 420 |

(net / 818 = full-test Δ vs primary: A −0.0024, B 0.0, C +0.0013, D 0.0.)

## Per-agent accuracy on the 84 escalated (deterministic capture)
> Per-agent numbers come from each design's temp-0 instrumentation re-run; gpt-4o-mini
> has small residual non-determinism, so a capture can differ from its headline run by
> ~1–2 borderline samples (B's capture matched net 0 exactly; C's and D's captures drift
> ≤1 sample). They characterise agent behaviour; the **net/accuracy rows above are the
> official headline runs.**

| agent | A | B | C | D |
|---|---|---|---|---|
| lexical | 0.7262 | (abstains) | 0.7143 | 0.7262 |
| logic | **0.6905** | — | — | **0.7024** |
| polarity | — | 0.7262 | 0.7381 | 0.7381 |
| contextual | 0.7500 | 0.7381 | 0.7381 | 0.7381 |
| final consensus | 0.7262 | 0.7500 | 0.7381* | 0.7619* |
| (Ahmed primary) | 0.7500 | 0.7500 | 0.7500 | 0.7500 |

\* capture value; C/D official escalated acc = 0.762 / 0.750.

**Logic is the weakest wherever it appears** (A 0.690, D 0.702); **Polarity is
consistently ~0.726–0.738** — a clear upgrade over Logic; **Contextual is consistently
strong (~0.738–0.750).**

## Agent agreement / diversity
| pair / measure | A | B | C | D |
|---|---|---|---|---|
| all-agree | 84.5% | 91.7% | **81.0%** | 78.6% |
| lexical↔logic | 0.893 | — | — | 0.881 |
| lexical↔polarity | — | — | **0.833** | 0.845 |
| logic↔contextual | 0.893 | — | — | 0.893 |
| polarity↔contextual | — | 0.917 | 0.905 | 0.929 |
| lexical↔contextual | 0.905 | — | 0.881 | 0.869 |

Notes: **B's "all-agree" 91.7% is only 2 agents** (mechanically higher; it is just the
single Polarity↔Contextual pair, which is *highly* correlated → little independent
signal). **D's 78.6% is mechanically lower** (4 agents rarely all agree) but it
re-introduces the redundant Logic (Logic↔Lexical 0.881). **C has the most genuine
3-way diversity** — the lowest redundant pair in the whole study (Lexical↔Polarity
**0.833**).

---

## Answering the main question — best balance across six axes
| axis | best | reading |
|---|---|---|
| **final performance** | **C** | 0.9267 > B=D 0.9254 > A 0.9230 |
| **net agent effect** | **C** | +1 > B=D 0 > A −2 |
| **reduced harmful overrides (C→W)** | **B** | 10 < C=D 11 < A 14 |
| **agent diversity** | **C** | 3 distinct roles, lowest redundancy (Lex↔Pol 0.833); B too few agents & highly correlated; D re-adds redundant Logic |
| **lower literalism (breaks)** | **B** | 10 < C=D 11 < A 14 |
| **safe on a strong primary** | **B / D** | both recover primary exactly (net 0); C +1 (within noise); A −2 (unsafe) |

### What each design teaches
- **A (keep Logic): the only net-negative design.** Logic (0.690) is the weakest and most
  redundant agent; with two surface agents (Lexical+Logic, 0.89 correlated) the correlated
  bloc over-rides the strong primary → net −2. Confirms Logic is the wrong specialist.
- **B (Polarity + Contextual): the "do no harm" design.** Dropping *both* surface
  literalists leaves a disciplined Polarity decider + Contextual → **fewest harmful
  overrides (10) and fewest literalism breaks (10)**, recovering the primary exactly
  (net 0) at the **lowest cost ($0.035)**. But with only 2 highly-correlated voters
  (0.917) it also makes the **fewest rescues (10)** — it adds no value, it just stops
  subtracting it.
- **C (Lexical + Polarity + Contextual): the best overall.** Replacing only the weak Logic
  keeps three genuinely distinct roles (lowest redundancy in the study), giving the **best
  final accuracy (0.9267), the only net-positive result (+1), and the best escalated
  accuracy (0.762)** — at modest cost ($0.049), with **no consensus change** (drop-in 3
  votes).
- **D (4 agents): dominated.** Adding Polarity *on top of* Logic re-introduces the
  redundant weak agent; the 4-vote bloc nets **0 — same as B — but costs the most
  ($0.064, +30% over C)** and needs the consensus `polarity` slot wired in. No reason to
  prefer it: it buys nothing over B/C and is the most expensive and most complex.

### Ranking
**C > B > D > A** for a strong primary.
- **C** wins on performance, net effect, and diversity.
- **B** wins on raw safety/cost/literalism but adds nothing (no rescues).
- **D** is strictly dominated (= B's net at higher cost + complexity).
- **A** is the only harmful design.

---

## Caveats (do not overclaim)
- **The spread is within run-to-run noise.** B/C/D all sit within **±1 sample** of
  primary_only (0.9254); C's +1 is **+0.0013 / one sample**, and temp-0 captures drift
  ~1–2 samples. The robust statement is: **A is clearly the worst; B, C and D all
  neutralise the agentic harm and land at/above the primary; C is the best point
  estimate.** It is **not** a statistically firm ordering among B/C/D.
- This is a **strong-primary** regime where the agent ceiling (~0.73–0.75) sits at/below
  the primary, so by design no configuration can beat primary_only by much. The decisive
  test of *value* is the **weak-primary (C3) regime**, where the agentic layer earns its
  keep — not run here.
- Per-agent rows are from instrumentation re-runs (≤1–2 sample drift), not the official
  headline transition counts.

## Recommendation
- **Keep Design C as the lead experimental variant** (`--sentiment_agent_variant
  lexical_polarity_contextual`): best performance + diversity, no consensus change, modest
  cost. Do not promote to default yet (gain within noise; C3 regime untested).
- **Design B is the conservative fallback** if "never net-negative + lowest cost +
  fewest harmful overrides" is valued over the small upside — but note it adds no rescues.
- **Retire Design D** for sentiment: it is dominated (no gain over B, highest cost/
  complexity, re-adds the weak Logic).
- **Next:** validate **C** (and optionally B) on the **C3 generated-primary** full_agentic
  to confirm the polarity decomposition preserves the +0.059 weak-primary gain. Not run
  here, per instruction.

## Artifacts
- A: `experiment_ahmed_semantic_v1/` · B: `experiment_ahmed_designB_polarity_contextual/`
  · C: `experiment_ahmed_polarity/` · D: `experiment_ahmed_designD_four_agent/`
  (each: `full/` headline metrics+predictions+llm_usage, `error_attribution/` per-agent).
- Capture driver: `scripts/ahmed_design_attribution.py` (B/D),
  `scripts/ahmed_polarity_attribution.py` (C), `scripts/ahmed_semantic_v1_attribution.py` (A).
- Implementation/how-to-enable: `EXPERIMENT_SENTIMENT_POLARITY_AGENT_CHANGELOG.md`.
