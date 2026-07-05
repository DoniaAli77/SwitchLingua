# sequential_sentiment_v2 — Forward-Pragmatics Redesign (Design Only)

**Design only — no code, no runs, no LLM calls.** Redesigns the staged sentiment
pipeline so pragmatic reasoning happens **before** the polarity decision, as **structured
features**, rather than **after** it as a keep/revise review. Motivated directly by
`EXPERIMENT_STAGE3_PRAGMATIC_REDUNDANCY_ANALYSIS.md`: v1's Stage-3 verifier was
confirmation-anchored (revised 3/84) because it received the proposed label and could only
ratify or overturn it. General sentiment analysis; no dataset/author/benchmark named; every
stage JSON-only with a confidence. Date: 2026-07-01.

## Core idea
```
v1 (review):   text → Intent → Polarity(label) → Pragmatic REVIEW(keep/revise) → controller
v2 (forward):  text → Intent → Pragmatic FEATURES(no label) → Polarity(decide once, feature-aware) → controller
```
The final polarity decision is made **once**, with intent + pragmatic features in context,
and is **never shown a prior label to defend.** Pragmatics is an *input*, not an *audit*.

---

## 1. Stage prompts & JSON schemas

### Stage 1 — Intent / opinion-expression detector (lean gate)
Keeps only the cheap opinion-existence judgment; the detailed pragmatics move to Stage 2.

**System prompt (essentials):**
```
You decide ONE thing: does the AUTHOR express their own evaluative opinion in this short
message? You do NOT assign sentiment polarity and you do NOT analyze sarcasm here.
  opinion_expressed: true  = the author gives their own evaluation (like/dislike, praise/criticism)
                     false = no author evaluation (neutral question, factual/plot description,
                             relayed/quoted opinion, request/advice, or a bare mention/reference)
                     unclear = genuinely ambiguous
Base it on what the author is DOING, not on the presence of emotional words. Handle
code-switched / mixed-language and informal text. Respond with ONLY JSON, exactly these keys.
```
**Schema:**
```jsonc
{"opinion_expressed": true|false|"unclear",
 "target": "<string or null>",     // coarse: what the message is about
 "confidence": 0.0,                 // certainty in opinion_expressed
 "evidence": ["<span>"]}
```

### Stage 2 — Pragmatic feature extractor (NO sentiment label)
Emits the structured pragmatic decomposition. **It must not output positive/negative/neutral.**

**System prompt (essentials):**
```
You extract structured PRAGMATIC FEATURES of a short message for a downstream sentiment
decision. You do NOT output a sentiment label (positive/negative/neutral) — only features.
Determine:
- speech_act: evaluate | describe | ask | advise | quote | other
- target: what any stance is about (string or null)
- target_attribution: author | other | none  (whose stance is it — the author's, someone
  else's that is being reported, or none)
- use_vs_mention: use | mention | platform_meta  (is emotional/entity language USED to
  evaluate, merely MENTIONED/named, or talk ABOUT the platform/interface/actions)
- platform_meta: true|false  (talk about likes/blocks/follows/comments/trending/clips/lyrics)
- description_vs_evaluation: evaluation | description | mixed
- sarcasm_or_irony: true|false  (does the author mean the OPPOSITE of the literal words)
- implicit_stance: positive | negative | none  (an implied stance not stated outright)
- stance_strength: none | weak | moderate | strong
Judge the AUTHOR, not the sentiment of a quoted/mentioned/described thing. Handle
code-switched/informal text. Respond with ONLY JSON, exactly these keys.
```
**Schema:**
```jsonc
{"speech_act": "evaluate|describe|ask|advise|quote|other",
 "target": "<string or null>",
 "target_attribution": "author|other|none",
 "use_vs_mention": "use|mention|platform_meta",
 "platform_meta": true|false,
 "description_vs_evaluation": "evaluation|description|mixed",
 "sarcasm_or_irony": true|false,
 "implicit_stance": "positive|negative|none",
 "stance_strength": "none|weak|moderate|strong",
 "confidence": 0.0,                 // certainty in the feature set
 "evidence": ["<span>"]}
```
*(`platform_meta` is intentionally derivable from `use_vs_mention=="platform_meta"`; kept as
a discrete boolean because the controller keys on it directly.)*

### Stage 3 — Polarity resolver (decides ONCE, feature-aware; sees NO prior label)
**System prompt (essentials):**
```
You assign the final sentiment polarity, using the message, an INTENT judgment, and a set
of PRAGMATIC FEATURES. This is the ONLY step that outputs a label. You are NOT reviewing a
previous label — decide fresh, informed by the features:
- If the features indicate no author evaluation (opinion_expressed false, use_vs_mention
  mention/platform_meta, description_vs_evaluation description, implicit_stance none) →
  usually "neutral".
- If sarcasm_or_irony is true, the intended stance is typically the OPPOSITE of the literal
  wording — resolve to the intended polarity, not the literal one.
- Otherwise resolve positive/negative/neutral from the author's expressed or implicit stance,
  weighting stance_strength; handle negation, intensifiers, and mixed polarity (dominant
  side; if truly balanced with no dominant side → neutral).
Judge the AUTHOR's stance. Use the features as evidence, but you may disagree with them if
the text plainly contradicts a feature. Respond with ONLY JSON, exactly these keys.
```
**Schema:**
```jsonc
{"label": "positive|negative|neutral",
 "confidence": 0.0,
 "used_features": ["sarcasm_or_irony", "implicit_stance", ...],  // which features drove it
 "reasoning": "<one or two sentences>",
 "evidence": ["<span>"]}
```
The allowed-label list is injected verbatim into the Stage-2/Stage-3 user prompts (label-
generic, as in v1).

## 2. Controller rules (deterministic; simpler than v1)
The label now comes straight from Stage 3; the controller only (a) applies the no-opinion
neutral safety gate cross-checked against pragmatic features, and (b) handles weak/conflict
fallback. **There is no keep/revise, no `pragmatic_revision` rule** — sarcasm/mention are
resolved *inside* Stage 3.

```
constants: TAU_INTENT=0.60, TAU_LOW=0.45, USE_PRIMARY_FALLBACK=True

1. NO-OPINION NEUTRAL (safety gate)
   if intent.opinion_expressed == false and intent.confidence >= TAU_INTENT
      and pragmatic.implicit_stance == "none"
      and (pragmatic.use_vs_mention != "use" or pragmatic.description_vs_evaluation == "description"):
        final = neutral                      decided_by = "intent_no_opinion"
   # fires only when intent AND pragmatic features AGREE there is no evaluation → no
   # single-stage error can force neutral (cross-checked gate; cascade guard).

2. TRUST FEATURE-AWARE POLARITY
   elif polarity.confidence >= TAU_LOW:
        final = polarity.label               decided_by = "polarity_feature_aware"

3. WEAK / CONFLICTED FALLBACK
   else:  # polarity.confidence < TAU_LOW
        if USE_PRIMARY_FALLBACK and primary usable:
              final = primary.label          decided_by = "fallback_primary"
        else: final = polarity.label         decided_by = "fallback_polarity"
```
Primary participates only as router (unchanged) + safe fallback — never a voter.
Malformed-JSON handling identical to v1 (one retry per stage → safe coerced default;
degrade toward Polarity → primary → neutral). Persist the full trace (`intent`,
`pragmatic`, `polarity`, `decided_by`, `used_features`, thresholds) under
`state.extras`, and **serialize per-stage confidences** (the calibration gap).

## 3. How v2 differs from v1
| | v1 | v2 |
|---|---|---|
| Pragmatics position | **after** polarity (review) | **before** polarity (features) |
| Final decider input | text + intent (Stage 2 sees no pragmatics) | text + intent + **pragmatic features** |
| Stage 3 role | verifier: keep/revise a **shown** label | resolver: decide **once**, no prior label shown |
| Sarcasm/mention handling | a late keep/revise (fired 3/84) | an **input feature** to the one decision |
| Controller | had a `pragmatic_revision` rule | **no revision rule**; gate + trust + fallback |
| Anchoring risk | high (label + its evidence shown) | **removed** (no label to defend) |

## 4. Which v1 failure it addresses
The exact one diagnosed: **Stage 3's confirmation anchoring.** v1 handed the final stage a
proposed label with its reasoning, so it ratified by default (81 keep / 3 revise) — a
near-no-op validator. v2 **never shows a prior label to the final decider**, so there is
nothing to anchor to; pragmatic judgments (sarcasm, mention, implicit stance) enter as
*features the decision is built from*, not as an *audit applied afterward*. This converts
the pragmatic contribution from a rarely-fired override into an always-present input.

## 5. Which errors it might fix
On the escalated subset, the classes where v1's inert Stage 3 left value on the table:
- **Sarcasm / irony** (literal-inverting): now `sarcasm_or_irony=true` is in front of the
  decider *before* it commits, instead of a rare after-the-fact flip.
- **Mention / platform-meta neutrals**: `use_vs_mention` + `platform_meta` +
  `description_vs_evaluation` feed both the neutral gate and Stage 3 — the express-vs-mention
  axis (the one lever with signal) is now first-class in the decision, not just the gate.
- **Implicit stance under a weak literal read**: `implicit_stance` + `stance_strength` let
  Stage 3 commit to a polarity the literal wording underspecifies, without needing a
  separate review pass.

## 6. Risks that remain
- **The strong-primary ceiling is untouched.** v2 changes *how* the escalated decision is
  formed, not the information content of the escalated subset. On Ahmed it will very likely
  land at the ceiling again (v1 already reproduced primary-on-escalated; the parallel
  variants did too). **v2's plausible payoff is on a weak primary, not the strong one.**
- **Feature neglect.** Stage 3 may underuse the provided features (LLMs don't always
  condition faithfully on structured inputs) — mitigated by `used_features` (observable) but
  not eliminated. If neglected, v2 degrades toward "Stage 3 alone" = a single polarity call.
- **Cascade from Stage 2.** Wrong pragmatic features can mislead Stage 3 (e.g. a false
  `sarcasm_or_irony=true` inverts a correct read). v1's review at least couldn't invert a
  correct label as easily; v2 gives Stage 2 more upstream influence. The cross-checked
  neutral gate (Rule 1 requires intent AND pragmatic agreement) contains only the neutral
  case, not the polarity-inversion case.
- **Still one base model on one text** → the decision remains correlated with itself; v2 is
  not an ensemble and does not add independent evidence (by design — that would rebuild
  parallel voting). So its ceiling is the model's own reading ability.
- **Cost/latency unchanged**: still 3 serial LLM calls on escalated (~$0.05/run scale).
- Single temp-0 draw noise (±1–2 samples) applies to any eval.

## 7. Closer to Ahmed's forward-decomposition? — **Yes, materially closer than v1.**
Ahmed's pattern is a forward chain where each prompt **solves a fresh sub-question and adds
information**, and **no stage re-judges a prior prediction.** v2 is exactly that: Intent →
Pragmatic features → a single Polarity decision, each consuming upstream information, none
reviewing a downstream label. v2 **removes the one un-Ahmed-like step in v1** (the
self-review Stage 3). It is the first version whose *entire* control flow matches Ahmed's
evidence-accumulating decomposition — which is the strongest conceptual argument for it.

## 8. Test on C3 first or Ahmed first? — **C3 first. Unambiguously.**
Three independent lines now show the **strong Ahmed primary is at ceiling** (root-cause
equation; parallel ablations; and sequential_v1 reproducing primary-on-escalated at 62–63/84
faithfully). A better internal reasoning topology cannot manufacture signal that isn't in
the escalated subset, so an Ahmed run of v2 would most likely reproduce the ceiling again —
informative only as a null. **The weak C3 generated primary is the single regime with
headroom** (primary term small, escalated set larger and more recoverable): baselines
primary_only 0.6956/0.6830, full_agentic 0.7543/0.7387. Run **v2 on C3 first**; only if it
beats full_agentic there does an Ahmed confirmation run become worthwhile (and there mainly
as a robustness/stability check, not an accuracy bet). Include the pure-sequential ablation
(`use_primary_fallback=False`) and serialize per-stage confidences + `used_features` so the
feature-neglect and cascade risks in §6 are measurable from the first run.

## Bottom line
v2 fixes the *specific* mechanism that made v1 inert — it removes the confirmation anchor by
moving pragmatics upstream as features and deciding polarity once — and it makes the whole
pipeline a clean forward decomposition in Ahmed's style. **But it does not change the
strong-primary ceiling;** its testable upside is on the weak C3 primary, where pragmatic
features can actually move escalated decisions. Design is ready for a future minimal
implementation; **no code or runs performed.**

## Artifacts / basis
- `EXPERIMENT_STAGE3_PRAGMATIC_REDUNDANCY_ANALYSIS.md` (the anchoring/redundancy finding).
- `EXPERIMENT_SEQUENTIAL_SENTIMENT_V1_AHMED_RESULTS.md` + `decision_trace/` (v1 at ceiling,
  81 keep / 3 revise, faithful/no-coercion).
- `EXPERIMENT_SEQUENTIAL_SENTIMENT_V1_PROMPT_CONTROLLER_DESIGN.md` (v1 baseline design).
- `EXPERIMENT_CONSENSUS_INVESTIGATION_SUMMARY.md` (express-vs-mention is the only lever).
