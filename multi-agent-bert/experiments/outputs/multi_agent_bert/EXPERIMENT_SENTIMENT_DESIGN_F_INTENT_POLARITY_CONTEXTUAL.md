# Design F — Intent + Polarity + Contextual (remove Lexical)

Does the Lexical agent contribute genuine explicit-cue evidence, or mainly surface-cue
bias? Test: **remove Lexical** and run Intent + Polarity + Contextual on the Ahmed frozen
primary (semantic_v1 prompts, threshold 0.7, Fix-2 consensus w_primary 1.0, gpt-4o-mini,
no training/generation). Opt-in variant `intent_polarity_contextual`; Lexical abstains
(no LLM call, no vote); Polarity in the logic slot; Intent as the 4th-slot agent → 3 active
votes, existing consensus. Date: 2026-07-01.

## Setup
- Frozen primary = Ahmed aligned predictions (`precomputed`), EESA test (818).
- Agents: **Intent + Polarity + Contextual** (Lexical abstains). Only change vs C is
  Lexical → Intent (i.e. drop the explicit-evidence agent, add the intent detector).

---

## 1–9. Headline metrics vs C / E / B / primary_only
| design | acc | macro F1 | wtd F1 | esc-acc | W→C | C→W (harmful) | **net** | neu→neg / neu→pos | neg→neu / pos→neu | total breaks | cost/calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ahmed **primary_only** | **0.9254** | 0.9207 | 0.9254 | 0.750 | — | — | — | — | — | — | — |
| **B** Pol+Ctx | 0.9254 | 0.9212 | 0.9254 | 0.750 | 10 | 10 | 0 | 4 / 1 | 3 / 2 | 10 | $0.035 |
| **C** Lex+Pol+Ctx | **0.9267** | 0.9226 | 0.9266 | 0.762 | 12 | 11 | **+1** | 4 / 2 | 3 / 2 | 11 | $0.049 |
| **E** Lex+Intent+Pol+Ctx | 0.9267 | 0.9227 | 0.9266 | 0.762 | 11 | 10 | +1 | 4 / 2 | 3 / 2 | 10 | $0.064 |
| **F** Intent+Pol+Ctx | **0.9218** | **0.9180** | 0.9219 | **0.7143** | 11 | **14** | **−3** | **4 / 1** | **4 / 5** | **14** | $0.049 |

(items 1–9 for F: acc **0.9218**, macroF1 **0.9180**, wtdF1 **0.9219**, esc-acc **0.7143**,
W→C **11**, C→W/harmful **14**, net **−3**, neu→neg **4**, neu→pos **1**.)

**F is the worst design in the whole study — net −3, below even A (−2).** The failure
signature is **over-neutralization**: the polar→neutral breaks explode — **positive→neutral
2→5** and **negative→neutral 3→4** (total breaks 11→14) — while the neutral-side breaks stay
low (neu→pos falls to 1).

## 10–12. Per-agent capture (84 escalated, deterministic re-run)
> ≤1-sample temp-0 drift: the capture landed net −2 (final-esc 0.7262) vs the headline net
> −3 (0.7143). Per-agent rows characterise behaviour; §1's counts are the headline run.

| agent | accuracy on escalated |
|---|---|
| Lexical | (abstains — no vote) |
| polarity | 0.7143 |
| **contextual** | **0.7500 (strongest)** |
| intent | 0.7143 |
| final consensus | 0.7262 (capture) / **0.7143 (headline)** |
| (Ahmed primary) | 0.7500 |

| agreement | value |
|---|---|
| all-3-agree | **76.2%** (64/84) |
| intent ↔ polarity | 0.8333 |
| intent ↔ contextual | 0.7738 |
| polarity ↔ contextual | 0.9167 |

- **Contextual is again the strongest agent (0.750)**; Polarity and Intent tie at 0.714.
- With no Lexical anchor, the three deciders agree less overall (76.2%) but all lean the
  *same direction* (neutral) on genuinely evaluative text — so the panel loses polar cases.

## Interpretation
- **Design F shows Lexical is genuinely useful and must not be removed.** Removing it
  caused **over-neutralization**: positive→neutral and negative→neutral breaks increased
  sharply (2→5 and 3→4). Intent + Polarity + Contextual, all neutral-leaning by design,
  became **too conservative without an explicit lexical-evidence anchor** to say "a real
  positive/negative cue is present here."
- **This confirms the best sentiment decomposition is C = Lexical + Polarity + Contextual**
  (net +1) — F's −3 is the direct cost of dropping the evidence agent.

### Connection to Ahmed's prompt/feature strategy (see `EXPERIMENT_AHMED_PROMPT_CLUES_AGENT_MAPPING.md`)
Ahmed's thesis makes this mechanistic, not incidental:
- **His model used a sentiment lexicon (2,324 phrases) + 19 handcrafted word features**
  (pos/neg/neutral, negation, intensifier, emoticon, repeated-character, named-entity, …)
  and **fine-tuning sentiment hints** — and these **always help** (GPT-4o-mini 93.77% →
  95.48% *with* hints). Explicit evidence is load-bearing in every strong Ahmed
  configuration → **supports keeping a Lexical / evidence agent.** Removing it (F) moves
  away from Ahmed's design and reproduces exactly the degradation his ablations predict.
- **His Rule-Based Inference prompt replaces broad "logic" with a polarity+intent
  decision** (find pos/neg phrases → decide polarity; negation flips, both-polarities ⇒
  sarcasm ⇒ negative) → **supports replacing the broad Logic agent with Polarity** (our
  Design C), not deleting Lexical.
- **His Semantic-Translation / whole-message prompting** → **Contextual remains useful**
  for whole-message interpretation (and F confirms Contextual is still the strongest agent,
  0.750).
- **Intent is useful conceptually** (Ahmed's author-vs-reader guideline and the
  "dislikes-question = neutral" rule are pure intent detection) **but should not replace
  Lexical, and should not be a co-equal voting agent** — as a neutral-leaning vote it
  drags polar cases to neutral (F) and adds nothing net as a 4th vote (E). It is better
  cast as a **gate** (Ahmed uses intent as a branch, not a vote — see the mapping report).

## Final architecture recommendation
**C = Lexical + Polarity + Contextual.**
- Lexical = explicit evaluative-cue evidence + strength (Ahmed's lexicon/hints/features).
- Polarity = the polarity decision incl. negation/intensifier/sarcasm (Ahmed's Rule-Based
  polarity step, replacing broad Logic).
- Contextual = whole-message / implicit / sarcasm interpretation (Ahmed's semantic
  translation).
- Intent = keep **conceptually**, but **not as a voter**; revisit only as an **Intent-as-gate**
  redesign (design-only; not implemented; not to be run before the pending C3 check).

Retire B/D/E/F as production candidates for sentiment (B safe-but-adds-nothing;
D dominated; E ties C at higher cost; **F actively harmful**).

## Caveats
- All designs sit within ±1–2 samples of primary_only *except A and F*, which are clearly
  below it (A −2, **F −3**). F's harm is the largest agentic effect measured and is robust
  in sign (headline −3, capture −2 — both negative), unlike the ±1 noise separating B/C/E.
- Per-agent rows are from the instrumentation re-run (≤1-sample drift), not the official
  headline transition counts.

## Artifacts
- Headline: `experiment_ahmed_designF_intent_polarity_contextual/full/` (metrics,
  predictions, `__llm_usage.json`, `run.log`).
- Per-agent capture: `…/error_attribution/attribution_table.{csv,json}`; driver
  `scripts/ahmed_designF_attribution.py`.
- Comparators: C `experiment_ahmed_polarity/`, E `experiment_ahmed_designE_intent/`,
  B `experiment_ahmed_designB_polarity_contextual/`.
- Ahmed mapping: `EXPERIMENT_AHMED_PROMPT_CLUES_AGENT_MAPPING.md`.
