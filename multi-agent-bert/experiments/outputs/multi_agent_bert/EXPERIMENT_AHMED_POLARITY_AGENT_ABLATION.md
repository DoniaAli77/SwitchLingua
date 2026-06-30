# Design C — Ahmed Frozen-Primary Full-Agentic with the Polarity Agent (Lexical + Polarity + Contextual)

Architectural ablation: replace the weak **Logic** agent with a sentiment-specific
**Polarity** agent, keeping Lexical and Contextual, on top of the `semantic_v1` prompts.
**Only the agent trio changed** (`--sentiment_agent_variant lexical_polarity_contextual`,
with `--sentiment_prompt_variant semantic_v1`); every other setting is identical to the
prior Ahmed runs. No training, no generation, no router/consensus/primary change.
Date: 2026-06-30.

## Setup (identical to B1, except Logic→Polarity)
- Frozen primary = Ahmed aligned predictions (`precomputed`), EESA test (818).
- active_task sentiment_classification · pipeline_mode full_agentic · threshold 0.7 ·
  Fix-2 consensus ON (w_primary 1.0) · agents_use_primary_signal false · gpt-4o-mini
  (temp 0.0) · `semantic_v1` prompts.
- **Only change:** agent trio = **Lexical + Polarity + Contextual** (Logic replaced). The
  Polarity agent writes the same `logic_output` slot, so **consensus is unchanged**.

---

## 1. Main metrics (full test, 818)
| configuration | accuracy | macro F1 | net (escalated) |
|---|---|---|---|
| Ahmed **primary_only** | **0.9254** | **0.9207** | — |
| old full_agentic (default prompts) | 0.9205 | 0.9153 | **−4** |
| `semantic_v1` (Lex+Logic+Ctx) | 0.9230 | 0.9183 | **−2** |
| **Design C polarity (Lex+Polarity+Ctx)** | **0.9267** | **0.9226** | **+1** |

weighted F1 (Design C) = **0.9266**. **Design C is the first agentic configuration whose
full-test accuracy exceeds Ahmed primary_only** (0.9267 > 0.9254).

## 2. Escalated subset (84)
| metric | value |
|---|---|
| escalated | **84 / 818 (10.3%)** |
| escalated-only accuracy (final) | **0.7619** (64/84) |
| wrong→correct | **12** |
| correct→wrong | **11** |
| **net** | **+1** |

This is the first time **escalated-only final accuracy (0.7619) exceeds the Ahmed primary's
own escalated accuracy (0.7500)** — the agents are, on net, *adding* correct answers on the
hard subset rather than removing them.

## 3. Error transitions (correct→wrong breaks, headline run)
| break (true→final) | old | semantic_v1 | **Design C** |
|---|---|---|---|
| neutral→negative | 7 | 5 | **4** |
| neutral→positive | 4 | 4 | **2** |
| negative→neutral | 2 | 3 | **3** |
| positive→neutral | 2 | 2 | **2** |
| **total breaks** | **15** | **14** | **11** |

The dominant surface-cue failure (neutral→polar) keeps shrinking: neutral→negative 7→5→**4**,
neutral→positive 4→4→**2**.

## 4. Per-agent capture (84 escalated, deterministic re-run)
> The per-agent diagnostics come from a separate temp-0 instrumentation re-run of the 84
> escalated samples (the pipeline persists only the final label). gpt-4o-mini has small
> residual non-determinism even at temperature 0: this capture landed **2 borderline
> true-neutral samples** differently from the headline run (`ahmed-eesa-00310`, `-00330`,
> both → positive in the capture, neutral in the headline), so the capture's net is **−1**
> vs the headline's **+1**. The capture is therefore representative for *agent behaviour*
> but is ~2 samples off the official transition counts in §2 (which are the headline run).

| agent | semantic_v1 (Logic) | **Design C** |
|---|---|---|
| lexical | 0.7143 | **0.7143** |
| **Logic → Polarity** | **0.690 (Logic)** | **0.7381 (Polarity)** |
| contextual | 0.7262 | **0.7381** |
| final consensus | 0.7024* | 0.7381 (capture) / **0.7619 (headline)** |
| (Ahmed primary) | 0.750 | 0.750 |

\* semantic_v1 final-consensus 0.7024 was measured on its own capture; Design C headline
final = 0.7619.

**Agreement (capture):**
| pair | semantic_v1 | **Design C** |
|---|---|---|
| all-3 agree | 84.5% | **81.0%** (68/84) |
| **Lexical ↔ Polarity** | 0.893 (Lex↔Logic) | **0.8333** |
| Polarity ↔ Contextual | 0.893 (Logic↔Ctx) | **0.9048** |
| Lexical ↔ Contextual | 0.905 | **0.8810** |

- **Is Polarity stronger than the old Logic? YES.** 0.690 → **0.7381** (**+0.048**) — the
  single biggest per-agent jump in the whole sentiment line of work. The replacement did
  exactly what it was designed to: the weakest specialist became a competent one.
- **Does Contextual remain strongest?** It is now **tied for strongest** with Polarity
  (both **0.7381**), both above Lexical (0.7143). So Contextual is no longer *uniquely*
  best — Polarity rose to match it. (Honest framing: Contextual did not weaken; Polarity
  caught up.)
- **Lexical ↔ Polarity = 0.833**, *below* the old Lexical↔Logic 0.893 and below the
  ≈0.9 "merge" trigger from the design proposal → **Lexical and Polarity are genuinely
  distinct; the C→B merge is not warranted.** Design C stands on its own.
- All-3 agreement fell further (84.5% → **81.0%**) — continued, intended decorrelation.

### Literalism — 2 more artifact breaks fixed vs semantic_v1
- **`ahmed-eesa-00239`** (true *neutral*): *"العربية الى بيتكلم عنها Dodg :D"* (the car
  he's talking about, Dodge :D). semantic_v1: lexical+logic both **positive** (brand name
  + `:D` emoji) → **positive ✗**. Design C: lexical still positive (weak cue) but
  **Polarity → neutral** (a brand mention + emoji is not an expressed evaluation) →
  **neutral ✓**.
- **`ahmed-eesa-00396`** (true *neutral*): *"Black widow يا دود"* (just naming a character).
  semantic_v1: lexical+logic **negative** → **negative ✗**. Design C: **Polarity → neutral**
  → **neutral ✓**.

Both are exactly the *mention-vs-expression* / *emoji-and-name-as-weak-cue* failures the
Polarity agent was defined to catch. 13 breaks remain (the indignant "unlike" meta-comments
and implicit insult/sarcasm cases — unchanged from semantic_v1).

## 5. Interpretation
- `semantic_v1` improved prompt-level reasoning (net −4 → −2, decorrelation 92%→84.5%) but
  **stayed below Ahmed primary_only** — better-worded agents were still the wrong *mix*.
- Replacing **Logic** with **Polarity** targeted the sentiment-specific failure mode
  directly: instead of generic structural reasoning, the second agent now answers the one
  question that matters here — *"does the author express an evaluative attitude, and if so
  what polarity?"* — with mention-vs-expression and weak-cue discipline built in.
- **Design C is the first agentic configuration to go net-positive on Ahmed** (net +1,
  0.9267 > primary_only 0.9254; escalated 0.7619 > primary's 0.7500). The agentic layer
  finally adds, rather than subtracts, value on a strong primary.
- The improvement is **small but meaningful as an architectural proof-of-concept**: it
  shows the bottleneck was not only prompt wording but the **specialist decomposition** —
  the right *set of roles*, not just better instructions for the existing roles.
- The evidence indicates **Logic was too broad / overlapping** for sentiment (weakest at
  0.690, ~0.89 correlated with Lexical), whereas **Polarity is a better sentiment
  specialist** (0.738, only 0.833 correlated with Lexical) — more accurate *and* more
  independent.

## 6. Caveats (do not overclaim)
- **The gain is very small: +1 sample / +0.0013 accuracy.** It is within the run-to-run
  noise of the system: the deterministic capture of the *same* config landed at **net −1**
  (0.9242), i.e. two borderline samples flipped between two temp-0 runs. The true effect is
  best described as **"Design C is on par with / marginally above Ahmed primary_only, and
  clearly better than semantic_v1 and the original prompts."** It is **not** a robustly
  demonstrated win over the primary.
- **Robustness is unconfirmed.** A claim of "beats the primary" needs a **repeat run and/or
  temperature/seed-stability check** (e.g. 3 runs) given the ±2-sample variability observed.
- **The per-agent numbers are from the instrumentation run** (net −1), ~2 samples off the
  headline (net +1); they characterise agent behaviour, not the exact official transitions.
- **Generated-primary regime untested.** Design C must still be checked on the **C3
  generated-primary** full_agentic to ensure it does **not** erode the previous **+0.059**
  weak-primary gain (the regime where the agentic layer earns most of its value).
- The agent-ceiling conclusion is unchanged in spirit: on a strong primary the headroom is
  tiny; Design C converts a small loss into a small (noisy) gain — it does not lift the
  agents far above the ~0.75 ceiling.

## 7. Recommendation
**For sentiment:**
- **Keep Design C as an experimental variant** (`--sentiment_agent_variant
  lexical_polarity_contextual`); it is opt-in and default behaviour is unchanged.
- **Do not replace the default trio yet** — the gain is within noise and unconfirmed on the
  generated-primary regime.
- **Next validation: C3 generated-primary full_agentic with the same polarity variant**,
  after this report — to confirm Design C preserves the +0.059 weak-primary gain. (Not run
  now; awaiting approval.) A temperature/seed-stability repeat of the Ahmed run would
  further firm up the +1 result.

## Artifacts
- Headline run: `experiment_ahmed_polarity/full_agentic_th07_polarity/` (metrics,
  predictions, `__llm_usage.json` = 336 calls / $0.0494, `run.log`).
- Per-agent capture: `experiment_ahmed_polarity/error_attribution/attribution_table.{csv,json}`
  + `error_attribution_capture.log`; driver `scripts/ahmed_polarity_attribution.py`.
- Comparators: `experiment_ahmed_semantic_v1/` (semantic_v1) and
  `experiment_ahmed_frozen_primary/` (original prompts).
- Implementation + how-to-enable: `EXPERIMENT_SENTIMENT_POLARITY_AGENT_CHANGELOG.md`.
