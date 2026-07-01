# Design v3 — Pragmatic-Reasoner Contextual (in-place Contextual upgrade)

Upgrades **only the Contextual agent** to an explicit Pragmatic Reasoner (speech act ·
target attribution · mention-vs-use · implicature/sarcasm · description-vs-evaluation →
then sentiment). Prompt-only, JSON schema unchanged, one vote — no new agent, no 4th
voter, router/consensus/Lexical/Polarity/IntentGate all unchanged. Architecture =
**Lexical + Polarity + Pragmatic-Contextual + IntentGate** (= Design G's agent set with the
Contextual prompt swapped). Same Ahmed frozen-primary setup (threshold 0.7, w_primary 1.0,
gpt-4o-mini). Opt-in `--sentiment_prompt_variant semantic_v3_pragmatic_contextual`; default,
semantic_v1, and G unchanged. Date: 2026-07-01.

---

## 1–7, 11, 12. Headline metrics vs primary_only / C / G
| design | acc | macro F1 | wtd F1 | esc-acc | W→C | C→W | **net** | neu→neg / neu→pos | neg→neu / pos→neu | breaks | cost/calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ahmed primary_only | 0.9254 | 0.9207 | 0.9254 | 0.750 | — | — | — | — | — | — | — |
| C | 0.9267 | 0.9226 | 0.9266 | 0.762 | 12 | 11 | +1 | 4 / 2 | 3 / 2 | 11 | $0.049 |
| **G** IntentGate | **0.9279** | 0.9242 | 0.9279 | 0.7738 | 10 | 8 | **+2** | 2 / 1 | **3** / 2 | 8 | $0.064 |
| **v3** pragmatic-Contextual + gate | **0.9279** | **0.9242** | 0.9280 | 0.7738 | 10 | 8 | **+2** | **1** / 1 | **4** / 2 | 8 | $0.067 |

**v3 ties G exactly** — same accuracy (0.9279), same net (+2), same 59/818 wrong, same
escalated accuracy. **Did not improve, did not reach 0.930.** The break *distribution*
shifted: **neutral→negative dropped 2→1** (the pragmatic reasoner correctly de-escalated a
meta-comment) but **negative→neutral rose 3→4** (one genuine negative over-neutralized) — a
one-for-one trade → net zero.

## 8. Contextual accuracy vs old Contextual — **it improved** (the key positive)
| agent (escalated) | G (semantic_v1 Contextual) | **v3 (pragmatic Contextual)** |
|---|---|---|
| lexical | 0.7381 | 0.7143 |
| polarity | 0.7381 | 0.7262 |
| **contextual** | **0.7381** | **0.7619** (+0.024) |
| intentgate | 0.7262 | 0.6905 |
| final consensus | 0.7857* | 0.7500* (capture; headline 0.7738 = G) |

\*capture values (±1–2-sample temp-0 drift); headline final = G.

**The pragmatic reasoning genuinely upgraded the Contextual *agent*: 0.7381 → 0.7619, now the
strongest agent and above the primary's 0.750.** But the improvement **did not propagate to the
final** — Contextual is 1 of 3 votes + gate, so a better Contextual vote is diluted/outvoted,
and it was offset by a new over-neutralization (see §14). Contextual changed its label on only
**4/84** escalated cases vs G — few changes, net-positive for the agent, net-zero for the system.

## 9. Agent agreement
all-3-agree **71/84 = 84.5%** (same as semantic_v1's 84.5%); pairwise lex-pol 0.869 · lex-ctx
0.905 · pol-ctx 0.917. The pragmatic prompt did **not** decorrelate the panel further — it
sharpened Contextual's *accuracy* without changing how often it agrees with the others.

## 10. IntentGate interventions
| | G | **v3** |
|---|---|---|
| gate said neutral | 44/84 | 45/84 |
| interventions | 6 | **7** |
| helped | 4 | **4** |
| hurt | 2 | **3** |

The gate fired once more and hurt once more (3 vs 2) — a v3×gate interaction: the stronger,
more-neutral-leaning pragmatic Contextual made one extra case look like an over-call the gate
then blocked. Not a gain.

## 13. Which pragmatic cases improved
- **Contextual component accuracy +0.024** — the pragmatic decomposition *did* improve the
  agent's own pragmatic judgements (it is now the best agent).
- **neutral→negative breaks 2→1** — one meta-comment that G's Contextual read as negative, the
  pragmatic Contextual correctly de-escalated (speech-act / mention-vs-use working).
- At the **final** level, however, **0 net new correct** (ties G) — the gains were absorbed.

## 14. Which cases regressed
- **`00113`** (true *negative*): *"زرع الزهرة وبعد تفتحها عاد وقطفها **wtf**"* ("planted the
  flower, then cut it, wtf"). G's Contextual → negative → correct. **v3's Contextual →
  neutral** (the *description-vs-evaluation* step (5) over-fired, reading it as recounting an
  event and missing the "wtf" evaluation) → final neutral → **wrong.** This is the
  over-neutralization risk (F-like) materialising in exactly one case — and it is what turned
  the neutral→negative gain into a wash (neg→neu 3→4).
- (Capture also showed `00100`/`00245` shifting, but those are gate/temp-0-noise interactions,
  not stable Contextual regressions.)

## 15. Cost
**420 calls / $0.0668** — slightly above G ($0.0636) because the pragmatic Contextual prompt is
longer (more prompt tokens per escalated sample).

---

## Interpretation
- **The Pragmatic Reasoner worked *as an agent upgrade* but not *as a system upgrade*.** It made
  the Contextual agent measurably better (+0.024, now the strongest agent), confirming that
  explicit pragmatic decomposition improves pragmatic judgement. But on a **strong** primary the
  system is already near its ceiling: a better Contextual vote is one of three, diluted by
  Lexical+Polarity and mediated by the gate, so it does not change the final tally.
- **The gain was offset by a new failure of the same kind it was meant to fix.** The
  description-vs-evaluation step, added to catch "recounting ≠ evaluating", over-applied on
  `00113` and flattened a real complaint — trading a recovered neutral→negative break for a new
  negative→neutral one. This is the F-like over-neutralization the design flagged as the main
  risk; it appeared, mildly (1 case).
- **Consistent with the gap analysis:** the reachable pool is fixed; a better pragmatic vote
  re-targets which cases are right without increasing the count, and 0.930 stays at the noise
  ceiling on this primary.

## Decision (against the stated criteria)
- "Improves G → new lead": **No** — ties G (0.9279, +2, 59 wrong).
- "Ties G → prefer simpler/safer *unless pragmatic errors clearly reduce*": pragmatic errors did
  **not clearly reduce** — Contextual accuracy rose and one meta break was fixed, but a new
  description-vs-evaluation over-neutralization appeared (net trade). → **prefer the simpler G.**
- "Over-neutralizes or hurts Contextual → retire": it **did not hurt Contextual** (Contextual
  improved), but it **did over-neutralize one case** (`00113`).

**Verdict: keep G as the lead for the strong primary; do not adopt v3 there** (ties on
accuracy, longer/complex prompt, one new over-neutralization). **But v3 is not a failure — it
is a genuinely better Contextual agent** whose benefit is masked on a near-ceiling strong
primary.

## Honesty / where v3 may actually matter
- On the **strong** Ahmed primary, v3 = G to the sample; any difference is within ±1–2-sample
  temp-0 noise. This is the expected near-ceiling outcome.
- **v3's real upside is a stronger Contextual vote, which should matter more on a *weak* primary
  (C3 generated)**, where the agents — not the primary — carry the decision and the agent-ceiling
  (~0.75) is the binding constraint. A Contextual at 0.76 instead of 0.74 has more leverage
  there. **This makes v3 (and G) both worth A/B-ing on C3** — the one decisive open test.
- Do **not** claim a 0.930 improvement from v3; there is none on the strong primary. The correct
  framing is: *the pragmatic upgrade improves the component and the pragmatic-error mix, but the
  strong-primary system is already at ceiling.*

## Recommendation / next
- **Lead stays G** on the strong primary. Retain v3 as an opt-in variant (it is a better
  Contextual and may help elsewhere); do not promote it to default.
- **Next (decisive, not run here): run C3 generated-primary full_agentic with (a) G and (b) v3**
  — to see whether the neutral-protecting gate erodes the +0.059 weak-primary rescue, and whether
  the stronger pragmatic Contextual adds more where the agents actually carry the load.

## Artifacts
- Headline: `experiment_ahmed_v3_pragmatic/full/`; capture: `…/error_attribution/…`; driver
  `scripts/ahmed_v3_attribution.py`.
- Prompt: `src/prompts/contextual_prompt.py` (`SYSTEM_PROMPT_PRAGMATIC`); design rationale
  `EXPERIMENT_PRAGMATIC_REASONER_DESIGN.md`; comparators G / `EXPERIMENT_G_TO_093_GAP_ANALYSIS.md`.
