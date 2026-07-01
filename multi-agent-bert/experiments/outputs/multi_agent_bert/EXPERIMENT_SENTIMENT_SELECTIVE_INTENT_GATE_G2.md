# Design G2 — Selective IntentGate (platform/meta only)

Can a **selective** IntentGate recover G's 2 gate-hurt cases without losing its 4 useful
platform/meta blocks? G2 = G with a prompt-only refinement: the gate returns **neutral
(→ triggers the neutral-guard) only for genuine platform/meta/mention/reference**, and
returns a **polar** label when the author expresses an implicit stance (insult, mockery,
fan excitement, informal praise/criticism, strong affect). Consensus guard logic unchanged;
the selectivity is entirely in the gate's prompt. Same Ahmed frozen-primary setup
(semantic_v1, threshold 0.7, w_primary 1.0, gpt-4o-mini). Opt-in
`lexical_polarity_contextual_selective_gate`; **G and default unchanged.** Date: 2026-07-01.

---

## 1–7, 12, 13. Headline metrics
| design | acc | macro F1 | wtd F1 | esc-acc | W→C | C→W | **net** | neu→neg / neu→pos | pos→neu / neg→neu | breaks | cost/calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ahmed primary_only | 0.9254 | 0.9207 | 0.9254 | 0.750 | — | — | — | — | — | — | — |
| C | 0.9267 | 0.9226 | 0.9266 | 0.762 | 12 | 11 | +1 | 4 / 2 | 2 / 3 | 11 | $0.049 |
| **G** IntentGate | **0.9279** | 0.9242 | 0.9279 | 0.7738 | 10 | 8 | **+2** | 2 / 1 | 2 / 3 | 8 | $0.064 |
| **G2** selective gate | **0.9279** | **0.9245** | 0.9279 | 0.7738 | 10 | 8 | **+2** | 3 / 2 | 2 / 1 | 8 | $0.064 |

**G2 ties G exactly** — same accuracy (0.9279), same net (+2), same 59/818 wrong, same
escalated accuracy (0.7738). macro F1 is a hair higher (0.9245 vs 0.9242, one recovered
negative), well within noise. **G2 did not reach 0.930 and did not improve over G.** The
break *distribution* shifted (neg→neu 3→1 recovered; neu→neg 2→3 / neu→pos 1→2 lost) but the
totals are unchanged — a pure reshuffle.

## 8–9. Gate interventions (capture) — more selective, same net
| | G | **G2** |
|---|---|---|
| gate said neutral (no-opinion) | 44/84 | **23/84** (much more selective, as intended) |
| interventions (blocked overrides) | 6 | **5** |
| helped | 4 | **4** |
| hurt | 2 | **1** |

The selective gate fires **half as often** and cut its hurt rate (2→1) — it *is* more
precise. But the reduced blocking also **dropped protections** it used to provide, so the
final tally is unchanged.

## 10. Are the 2 old gate-hurt cases recovered? — **1 of 2**
| case | G (gate/blocked/final) | G2 (gate/blocked/final) | true | recovered? |
|---|---|---|---|---|
| `00021` implicit insult ("…dور الممحونة… No one") | neutral / blocked / **neutral ✗** | **negative / not-blocked / negative ✓** | negative | **YES** — selective gate correctly saw the implicit insult |
| `00706` excited fan ("permission to dance… BTS…") | neutral / blocked / neutral ✗ | neutral / blocked / **neutral ✗** | positive | **NO** — gate still read it as content/brand *spotting* → neutral |

## 11. Are the 4 useful platform/meta blocks preserved? — **only 1 of 4**
| case | G | G2 | true | preserved? |
|---|---|---|---|---|
| `00258` "who are the 74 who disliked" (plain question) | neutral/blocked/neutral ✓ | neutral/blocked/**neutral ✓** | neutral | **YES** |
| `00203` "…على اساس اية!!" (indignant unlike-question) | neutral/blocked/neutral ✓ | **negative/not-blocked/negative ✗** | neutral | **NO** — read exclamatory affect as negative stance |
| `00245` "…ازااااى!!!" (indignant unlike-question) | neutral/blocked/neutral ✓ | **negative/not-blocked/negative ✗** | neutral | **NO** — same |
| `00320` BTS-logo spotting ("…🙂💜🌚") | neutral/blocked/neutral ✓ | **positive/not-blocked/positive ✗** | neutral | **NO** — read emojis as positive |

## 14. Cost
**420 calls / $0.0641** (same order as G; +30% over C).

## 15. Examples — the trade in one line each
- **Helped (recovered):** `00021` — the selective gate recognised the implicit insult (returned
  negative instead of neutral), so the guard did not block the agents' correct negative → **fixed**.
- **Lost (platform protections):** `00203`/`00245` — *indignant* unlike-questions ("on what
  basis?!", "how could they?!") carry exclamatory affect; the selective gate now labels them
  **negative** → guard doesn't fire → the agents' wrong negative override returns. `00320` — the
  gate read the emojis as **positive** → not protected.
- **Still lost:** `00706` fan cheer — the gate still classifies it as content/brand spotting
  (permission-to-dance / BTS references) → neutral → still blocks the correct positive rescue.

---

## Interpretation
- **The selective gate rebalanced but did not improve.** It is genuinely more precise (fires
  23 vs 44 times, hurt 1 vs 2), and it **recovered the clear implicit insult** (`00021`). But
  the same "return polar when affect is present" rule that recovered `00021` **also fired on the
  *indignant* platform questions** (`00203`/`00245`) — which are linguistically affective ("how
  could they?!") yet labelled neutral by convention (questions about *other users'* actions).
  So each recovered implicit case was offset by a lost platform case → **net zero**.
- **Root cause:** the boundary the gate must draw — *"affective platform question" (neutral) vs
  "expressed stance" (polar)* — is genuinely ambiguous, and the LLM cannot separate them cleanly
  in a single prompt. Making the gate strict enough to keep `00203`/`00245` (neutral) reopens
  the `00021` over-block; loosening it to fix `00021` drops `00203`/`00245`. **The two goals
  trade against each other.**
- This **empirically confirms the gap analysis**: the reachable escalated pool is a fixed set of
  ~19 hard, mostly-implicit/borderline cases, and re-targeting the gate **moves which ones are
  right without changing the count**. 0.930 is at the noise ceiling.

## Decision (against the stated criteria)
- "Reaches 0.930 **or** improves over G without losing platform/meta fixes → keep G2": **No** —
  G2 ties G **and lost 3 of 4 platform fixes**.
- "Ties G → prefer G only if simpler/safer": **G is simpler** (default gate prompt) **and safer**
  (its platform blocks align with the neutral-question convention). → **keep G.**
- "Loses the platform/meta protections → keep G": G2 **did** lose them → **keep G.**

**Verdict: retire G2; G (Lexical + Polarity + Contextual + default IntentGate) remains the lead
sentiment configuration.** The selective gate is a valid, more-precise idea but delivers no net
gain here because the implicit-stance carve-out and the platform-question protection are in
direct tension.

## Honesty / noise ceiling
- G2 = G to the sample (0.9279, +2, 59 wrong). The only difference is *which* ~2 escalated cases
  are right; the total is fixed. Any "+1/+2" from further gate re-targeting would be **within the
  ±1-sample temp-0 noise** already observed, not a robust gain.
- Consistent with the gap analysis: **0.930 with a frozen primary and general (non-tailored)
  changes is at/near the ceiling.** A robust >0.93 needs a stronger/better-calibrated primary
  (retraining) or EESA-style annotation-convention encoding (dataset-tailoring) — both excluded
  here. G is already at the practical ceiling.

## Recommendation / next
- **Lead stays G.** Do not adopt G2 (no gain, lost platform protections, more complex prompt).
- The remaining open question is unchanged: **validate G on the C3 generated-primary** (does the
  neutral-protecting gate erode the +0.059 weak-primary rescue?). That is the decisive test — not
  more gate re-targeting on the strong Ahmed primary, which is noise-bound.

## Artifacts
- Headline: `experiment_ahmed_designG2_selective_gate/full/`.
- Capture: `…/error_attribution/attribution_table.{csv,json}`; driver
  `scripts/ahmed_designG2_attribution.py`.
- Selective prompt: `src/prompts/intent_prompt.py` (`SYSTEM_PROMPT_SELECTIVE`).
- Comparators: `experiment_ahmed_designG_intent_gate/` (G),
  `EXPERIMENT_G_TO_093_GAP_ANALYSIS.md`, `EXPERIMENT_SENTIMENT_INTENT_GATE_ABLATION.md`.
