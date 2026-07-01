# What Stops Design G from Reaching 0.930? — Evidence-Gap Analysis

Analysis only — **no experiments, no prompt/code changes, no API calls.** Uses the saved
Design-G predictions (818) + per-agent/gate attribution table (84 escalated). Baselines:
Ahmed primary_only **0.9254**; best agentic **G 0.9279 / 0.9242 / net +2**; target **0.9300**.
Date: 2026-07-01.

## The arithmetic of the gap
- G is **59 / 818 wrong** (0.9279). **0.9300 = ≤ 57 wrong → we must fix just 2 more samples.**
- **But of the 59 wrong, 40 are NON-ESCALATED** — primary predictions with confidence ≥ 0.7
  that the router never sent to the agents. **The agentic layer literally never sees them.**
- Only **19 wrong are escalated** (agents ran): **8 harmful overrides (C→W)** + **11 missed
  rescues (W→W, primary already wrong)**. **This 19 is the entire pool the agents can affect.**

So the 2 samples needed for 0.93 must come from **either** shrinking the 19 escalated errors
**or** escalating+fixing some of the 40 invisible primary errors.

---

## 1. Remaining error analysis for Design G

### Confusion (true→pred), all 59 wrong
neutral→positive 17 · negative→neutral 18 · positive→neutral 13 · neutral→negative 7 ·
negative→positive 2 · positive→negative 2. **The errors are overwhelmingly *→neutral (31)*
and *neutral→ (24)* — i.e. the neutral↔polar boundary.**

### The 40 non-escalated (primary errors, agents blind)
Confident-but-wrong primary predictions (neg→neu 12, neu→pos 14, pos→neu 9, …). Sampled
texts are mostly **subtle/implicit**: complaints ("poor network stop us"), dark jokes,
"too good to be true" sarcasm, creepy-praise, copy-accusations, and a few platform/meta
("one dislike button?!", "whoever cuts their veins hits Like"). These are **genuine primary
mistakes the agentic layer cannot reach at threshold 0.7.**

### The 19 escalated errors (the only reachable pool) — typed
| # | sample | true→final | type | note |
|---|---|---|---|---|
| C→W | 00008 "…dة gay?" | neg→neu | **(b) implicit insult** | no explicit cue → all agents+gate neutral |
| C→W | 00097 "…no real man incl. Mohamed Ramadan 😂" | neg→neu | **(b) implicit mockery** | 😂 mockery, no lexical cue |
| C→W | 00193 "…واضح ان فيه cut…" | neg→neu | **(f) description-as-neutral** | complaint about editing read as description |
| C→W | 00290 "…ما كنت تخليك ف التمثيل أحسن 😏🙄" | neu→neg | **(g) borderline advice** | mild criticism; gate said "opinion", didn't block |
| C→W | 00307 "…JUST DO IT… 😄" | neu→pos | **(g) motivational/borderline** | arguably positive; gold neutral |
| C→W | 00635 "…مسلسل out of season" | pos→neu | **(b) implicit praise** | colloquial praise, no cue |
| C→W | 00642 "عاش عاش اسمك ايه في لعبة Free Fire" | pos→neu | **(b) implicit praise + (e) entity** | "bravo" + game-name question |
| W→W | 00021 "…دور الممحونة… No one : مى عمر" | neg→neu | **(b) implicit insult** | **gate HURT** (blocked correct rescue) |
| W→W | 00046 "اللي عاملين dislike… 😂😂" | neg→neu | **(a/b) meta + sarcasm** | primary wrong-neutral |
| W→W | 00182 "انت no one" | neg→neu | **(b) implicit insult** | no cue |
| W→W | 00220 "من يُحبنا ونحن في أسوأ حالاتنا…" | neu→pos | **(f/g) quoted aphorism** | wisdom quote read positive |
| W→W | 00240 "اخوانا اللى عاملين dislike على اساس ايه" | neu→neg | **(a) platform/meta** | **primary wrong-negative → gate can't protect** |
| W→W | 00250 "…Do not make idiots a role model" | neu→neg | **(g) advice** | advice = neutral by convention |
| W→W | 00295 "…Listen To My Remix: [link]…" | neu→pos | **(g) advertisement/self-promo** | ad = neutral by convention |
| W→W | 00298 "…ضل راجل=لايك الفتوة=dislike" | neu→neg | **(a) platform/meta** | primary wrong-negative |
| W→W | 00446 "Dislikes كتير اوي" | neu→neg | **(a) platform mention** | primary wrong-negative |
| W→W | 00542 "تول عومراك جاميلا XD XD" | pos→neu | **(b) misspelled/obscured praise** | + repeated-char |
| W→W | 00706 "يا ارمي… permission to dance… BTS…" | pos→neu | **(b) fan praise** | **gate HURT** (blocked rescue) |

**Type tally of the 19 reachable errors:**
- **(b) implicit sarcasm / insult / praise, no explicit cue — ~8 (the dominant cluster).**
- **(g) label-convention / borderline (ad, advice, motivational, quote) — ~5.**
- **(a) platform/meta where the *primary* is already wrong (so the gate, which protects a
  neutral primary, cannot fire) — ~3.**
- (f) description/quote — ~2 · (e) entity/target — ~1 (co-occurring).
- (c) negation, (d) emoji-as-sole-cause — **0 as a primary cause** (emojis co-occur but
  aren't the deciding error).

**Two structural facts:**
1. The gate already cleared the platform/meta cases where the **primary was neutral** (its 4
   "helped"). The remaining platform/meta errors (00240/00298/00446) have a **wrong-negative
   primary**, so the gate cannot help — a *router/primary* problem, not a gate problem.
2. The 2 **gate-HURT** cases (00021, 00706) are inside this pool — the gate over-fired on an
   implicit insult and a fan cheer. Un-gating exactly those two would recover 2 samples.

---

## 2. Map remaining errors → Ahmed's thesis components
| remaining error type | did Ahmed have a mechanism? | did it target this? |
|---|---|---|
| **(b) implicit sarcasm/insult/praise (no cue)** | annotation-guideline "sarcasm→negative" + semantic-translation (meaning-level) | **partially** — Ahmed's model still relies on cues; his *fine-tuned* model learned implicit cases from 2,464 labels (we can't fine-tune the frozen primary) |
| **(g) ad / advice / motivational = neutral** | **annotation-guideline prompt (explicit rules: ads, hyperlinks, advice, questions → neutral)** | **directly** — Ahmed encodes these as rules |
| **(a) platform/meta, primary wrong** | annotation-guideline "why-are-there-dislikes=neutral" + lexicon "dislike" as weak | targets the *text* but not a *wrong primary* |
| **(f) quote / plot description** | annotation-guideline (author-perspective) | partially |
| **(e) named-entity / target** | **named-entity feature** | yes (as a feature) |
| **negation / intensifier** | **negation & intensifier features** | yes — **but we have 0 such errors**, so no gain available |
| **emoji / repeated-char** | **emoji & repeated-char features** | yes — **but 0 as sole cause here** |
| **40 non-escalated primary errors** | preprocessing normalization + lexicon + **sentiment hints in fine-tuning** + ensemble | Ahmed fixed these **by training a better primary**, not by an agent layer |

**Key takeaway:** Ahmed's feature machinery (negation/intensifier/emoji/repeated-char/NE) maps
to error types we **do not actually have** in the reachable pool. His mechanisms that *do*
match our remaining errors are (i) the **annotation-guideline rules** (ad/advice=neutral —
but that's convention encoding) and (ii) **a better-trained primary** (sentiment-hint
fine-tuning) — which addresses the 40 invisible errors we can't reach without retraining.

---

## 3. Missing / underrepresented components in our pipeline
| Ahmed component | in G? | what it would add | prompt-only? | needs deterministic feature? | dataset-tailoring risk | ≥2-sample gain likely? |
|---|---|---|---|---|---|---|
| Sentiment lexicon / cue inventory | ~ (Lexical agent, no explicit lexicon) | stronger explicit-cue recall | yes | optional | low (generic lexicon) | **no** — our errors are cue-less |
| Negation / intensifier features | no | scope/flip handling | via prompt or feature | ideally feature | low | **no** — 0 such errors |
| Emoji / repeated-char features | ~ (prompts mention weak cues) | non-linguistic signal | yes | optional | low | **no** — 0 sole-cause |
| Named-entity feature | no | target attribution | feature | yes | low | marginal (1 case) |
| **Annotation-guideline rules (ad/advice/question→neutral)** | partial (gate = dislikes-question only) | fixes (g) cluster (ad/advice/motivational) | **yes** | no | **HIGH — encodes EESA label convention** | maybe 1–2, but tailored |
| **Semantic translation / meaning-level** | no | implicit meaning of code-switched text | yes (+1 call) | no | low (general) | **maybe 1–2** (implicit cases) |
| Rule-based inference (intent-gate) | **yes (Design G)** | already realized | — | — | — | already banked |
| Sentiment-hint fine-tuning of primary | **no (frozen primary)** | fix the 40 invisible errors | no | no | — | **yes but out of scope** (can't retrain) |
| Ensemble (Ahmed + XLM-R + agents) | no | — | no | no | low | **no** — XLM-R (0.85) << Ahmed (0.925), would hurt |

---

## 4. Ranked candidate improvements
| rank | change | expected benefit | risk | verdict |
|---|---|---|---|---|
| **1** | **C — More selective IntentGate** (fire only on platform/meta-comment patterns; do NOT gate implicit insults/fan-praise) | **+1 to +2** — directly recovers the 2 gate-HURT cases (00021 insult, 00706 fan) while keeping the 4 platform "helped" | low–med (could lose 00320 spotting) | **best bet for the 2 samples**; minimal, isolates cleanly, general (platform-meta is a generic category) |
| 2 | D — Semantic paraphrase/translation step before agents | +0–2 — may surface implicit code-switched insults/praise (the (b) cluster) | med (extra call/cost; translation can distort) | plausible, general, but effect uncertain and not targeted |
| 3 | A — Deterministic feature hints in prompt (negation/intensifier/emoji/platform-flag/NE/AR-EN ratio) | ~0–1 — our reachable errors are **cue-less**; negation/intensifier/emoji errors are 0 | low; but platform-flag overlaps the gate | Ahmed-faithful & general, but **mis-targeted** for the errors we actually have |
| 4 | B — FeatureHint/Evidence agent (deterministic hints, non-voting) | same as A, as a component | low–med (new component) | more infrastructure for the same low expected gain |
| 5 | E — Confidence calibration / non-overriding policy | ~0 accuracy (protective only) | low | doesn't add correct answers; we already protect via the gate |
| 6 | F — Ensemble Ahmed + XLM-R + agents | **negative** | — | **reject** — XLM-R 0.8533 ≪ Ahmed 0.9254; averaging in a weaker model lowers accuracy |
| — | "annotation-rule" ad/advice=neutral prompt | +1–2 but **dataset-tailoring** | **high** | encodes EESA labelling convention → excluded per the general-not-tailored rule |

---

## 5. Recommended next single change
**Refine the IntentGate to fire only on platform/meta-comment / mention patterns (Option C).**
- **Why it is most likely to gain the 2 needed samples:** the 2 samples are *literally* the 2
  gate-HURT cases (an implicit insult and a fan cheer) where the current gate over-fired on
  "no explicit opinion" and blocked a correct polar rescue. Restricting the gate to *clear
  platform/meta cases* (questions about likes/dislikes, mentions of UI actions) stops those 2
  bad blocks while retaining the 4 good platform blocks (the unlike/dislike-question cluster).
- **General, not dataset-specific:** "platform/meta-comment vs authored evaluation" is a
  generic pragmatic distinction (it is *not* an EESA label rule like "ads=neutral").
- **Minimal & isolable:** it tightens the gate's firing condition only; no new agent, no new
  vote, one guard predicate. Fully compatible with the current framework.
- **Expected effect:** net +2→ potentially **+3/+4 on the escalated subset → ~0.930–0.931** —
  *if* the two recovered rescues survive and no helped case is lost.

Second choice if C under-delivers: **D (a short semantic-normalization/translation step)** as
the only lever with a general shot at the implicit (b) cluster — but its effect is uncertain
and it adds cost.

---

## 6. Honesty / do-not-overclaim
- **The structural ceiling is the 40 non-escalated primary errors (68% of all G errors).**
  The agentic layer provably cannot reach them at threshold 0.7, and the earlier
  **router-selectability analysis proved a smarter router cannot select them either** (Ahmed's
  confidence is statistically identical on his right vs wrong escalated cases). Lowering the
  threshold to expose them would flood the agent layer (ceiling ~0.75) and **hurt** the ~694
  confident-correct primaries. So **0.93 cannot come from the primary-error bucket** without
  **retraining a better primary** (Ahmed's own route: sentiment-hint fine-tuning → 95.48%),
  which is out of scope for a frozen primary.
- **Within the reachable 19, the dominant remaining type is implicit, cue-less sarcasm/
  insult/praise (~8)** — the hardest category, where Ahmed himself succeeded only via
  *fine-tuning*, not prompting. General prompt/gate changes are unlikely to fix many of these.
- **Reaching exactly 0.930 is achievable but marginal and noise-adjacent.** It means fixing 2
  of 818 = +0.0025, and our temp-0 runs already drift ±1 sample. A gate-refinement (Option C)
  can plausibly land ~0.930, but it would be **squeezing 2 specific measured cases** — not a
  robust, comfortably-above-0.93 result, and it risks over-fitting to those 2.
- **A robust >0.93 realistically requires either (a) a stronger/better-calibrated primary
  (retraining with sentiment hints — Ahmed's actual 0.95 path) or (b) encoding EESA annotation
  conventions (ad/advice=neutral) — which is dataset-tailoring and excluded.** With a frozen
  primary and general-only changes, **~0.928–0.930 is approximately the ceiling**, and the
  honest expectation is that G is already near it.

## Artifacts
- G predictions: `experiment_ahmed_designG_intent_gate/full/…predictions.csv`.
- G attribution (per-agent + gate): `…/error_attribution/attribution_table.json`.
- Supporting: `EXPERIMENT_AHMED_ROUTER_SELECTABILITY.md` (primary errors unselectable),
  `EXPERIMENT_AHMED_PROMPT_CLUES_AGENT_MAPPING.md`, `EXPERIMENT_SENTIMENT_INTENT_GATE_ABLATION.md`.
