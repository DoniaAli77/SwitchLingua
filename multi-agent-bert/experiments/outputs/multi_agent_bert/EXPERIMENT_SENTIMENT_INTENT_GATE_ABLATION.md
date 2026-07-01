# Design G — IntentGate (Lexical + Polarity + Contextual + non-voting gate)

Does using **Intent as a gating guard** (not a voter) improve over Design C? The IntentGate
runs as a 4th agent but **casts no vote** (consensus polarity weight = 0); instead the
consensus applies a guard: if the agents overrode the primary but the gate **sides with the
primary**, the override is blocked. For sentiment this means *"gate judges no evaluative
opinion (neutral) + neutral primary → block the unsupported polar over-call."* Same Ahmed
frozen-primary setup (semantic_v1, threshold 0.7, Fix-2 w_primary 1.0, gpt-4o-mini, no
training/generation). Opt-in `lexical_polarity_contextual_intent_gate`. Date: 2026-07-01.

Gate behaviours implemented (all three from spec): (1) conservative neutral guard — block
polar override when gate = no-opinion and the primary is neutral; (2) the gate **never
overrides clear polarity by itself** (it only reverts overrides where it agrees with the
primary); (3) when the gate signals a clear opinion, normal Design-C consensus runs.

---

## 1–7, 10, 11. Headline metrics vs primary_only / C / E / F
| design | acc | macro F1 | wtd F1 | esc-acc | W→C | C→W | **net** | neu→neg / neu→pos | pos→neu / neg→neu (polar→neutral) | total breaks | cost/calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ahmed **primary_only** | 0.9254 | 0.9207 | 0.9254 | 0.750 | — | — | — | — | — | — | — |
| **C** Lex+Pol+Ctx | 0.9267 | 0.9226 | 0.9266 | 0.762 | 12 | 11 | +1 | 4 / 2 | 2 / 3 | 11 | $0.049 |
| **E** 4-vote | 0.9267 | 0.9227 | 0.9266 | 0.762 | 11 | 10 | +1 | 4 / 1 | 2 / 3 | 10 | $0.064 |
| **F** no-Lexical | 0.9218 | 0.9180 | 0.9219 | 0.714 | 11 | 14 | −3 | 4 / 1 | **5 / 4** | 14 | $0.049 |
| **G IntentGate** | **0.9279** | **0.9242** | **0.9279** | **0.7738** | 10 | **8** | **+2** | **2 / 1** | 2 / 3 | **8** | $0.064 |

(items for G: acc **0.9279**, macroF1 **0.9242**, wtdF1 **0.9279**, esc-acc **0.7738** (65/84),
W→C **10**, C→W **8**, net **+2**; neutral→negative **2** (was 4 in C), neutral→positive **1**;
polar→neutral breaks unchanged (pos→neu 2 / neg→neu 3).)

**G is the best design in the study** — highest accuracy (0.9279), highest escalated accuracy
(0.774), **lowest harmful overrides (C→W 8)**, **fewest total breaks (8)**, and the largest
positive net (+2). Crucially the gate did **not** raise polar→neutral breaks (unlike F's
over-neutralization) — it removed *neutral→polar* breaks (neu→neg 4→2) while leaving the
polar breaks alone.

## 8–9, 12. Gate interventions (from the deterministic capture)
- **Total gate interventions (blocked overrides): 6.**
- **Helped: 4** (blocked a *wrong* polar over-call of a correct-neutral primary → fixed a
  C→W break).
- **Hurt: 2** (blocked a *correct* polar rescue where the primary was wrong-neutral but the
  text truly was evaluative → lost a W→C rescue).
- **No correctness change: 0.**
- **Net gate effect ≈ +4 fixed − 2 lost.** That is exactly how C's +1 became G's +2: the gate
  cut C→W (11→8) by fixing the meta-comment cluster and cost only 2 rescues (W→C 12→10).
- The gate judged **"no expressed opinion" (neutral) on 44/84** escalated — it is active on
  ~half the hard set (which is dominated by questions / meta-comments), and it intervened
  only where the agents actually tried to override a neutral primary (6 of those 44).

**Per-agent accuracy (capture):** lexical 0.7381 · polarity 0.7381 · contextual 0.7381 ·
**IntentGate 0.7262** (as a standalone stance-labeler) · **final consensus 0.7857** (capture;
headline escalated 0.7738). The gate's own accuracy is the lowest — but it is **not voting**;
its value is the *guard precision* on interventions: **4/6 = 67% of its blocks were correct.**

## 13. Where the gate HELPED (all 4)
- **`ahmed-eesa-00203`** (true *neutral*): *"هو اللي عامل unlike دة عمله علي اساس اية!!"*
  ("the person who hit *unlike* — on what basis?!"), **`ahmed-eesa-00245`**: *"الناس الى
  عاملة unlike دى ازااااى"*, **`ahmed-eesa-00258`**: *"مين ال 74 ال عملين dislike"* ("who are
  the 74 who disliked"). In all three, **all three agents voted negative** (the exact
  "unlike/dislike" meta-comment literalism that stayed **broken in A / semantic_v1 / C / E**);
  the **gate said neutral** — these are *questions about other users' actions*, not the
  author's own evaluation — and blocked the override → **neutral ✓**. **This is the persistent
  still-broken cluster that no prior design fixed.**
- **`ahmed-eesa-00320`** (true *neutral*): BTS-logo *spotting* ("spot the shirt logo… 🙂💜🌚") —
  lexical+contextual read positive (brand + emojis); gate said neutral (a mention, not an
  evaluation) → **neutral ✓**.

## 14. Where the gate HURT (both 2)
- **`ahmed-eesa-00021`** (true *negative*): a crude insult; **primary was wrong-neutral**,
  the agents correctly overrode to negative, but the **gate mis-judged it as no-opinion** and
  reverted to neutral → **wrong**. A gate false-positive.
- **`ahmed-eesa-00706`** (true *positive*): an excited BTS fan; primary wrong-neutral, agents
  positive, gate reverted to neutral → **wrong**. The gate over-neutralized genuine enthusiasm.

Both hurt cases share the failure mode: **primary is wrong-neutral AND the text is genuinely
evaluative**, so protecting the primary loses a correct rescue. This is the intrinsic risk of
a primary-protecting gate — bounded here to 2/84.

## 15. Cost
**420 calls / $0.0636** (Lexical + Polarity + Contextual + IntentGate + explainability × 84).
Same order as E/D; +30% over C.

---

## Interpretation
- **IntentGate improves over C** on every headline metric (acc +0.0012, net +1→+2, escalated
  0.762→0.774) **and specifically reduces harmful overrides** (C→W 11→8) **without losing most
  rescues** (W→C 12→10). It does *not* over-neutralize like F — because Lexical + Polarity +
  Contextual still decide the polarity; the gate only vetoes *overrides that disagree with a
  neutral primary and are unsupported by expressed opinion*.
- **It fixes the one cluster nothing else could:** the "unlike/dislike/who-disliked"
  meta-comment questions. Every prior design (including C and E) got these wrong because all
  three voters read the surface "unlike/dislike" token as negative. The gate asks the right
  question — *is the author expressing their own evaluation?* — and answers *no* (it's a
  question about others), protecting neutral. This is exactly **Ahmed's annotation rule**
  (*"why are there dislikes" = neutral*) and his **Rule-Based Inference gate step** (*"is the
  author questioning the sentiment expressed by others? → neutral"*) realized as a guard.

## Decision (against the stated criteria)
- "Improves C **or** reduces harmful overrides without losing rescues → keep as candidate":
  **G does both** (net +1→+2; C→W 11→8; only −2 rescues) → **KEEP — G is the new best
  candidate.**
- It is **not** a wash like E and does **not** over-neutralize like F.

**Verdict: adopt G (Lexical + Polarity + Contextual + non-voting IntentGate) as the lead
sentiment configuration, pending the C3 generated-primary check.** The gate is the first
mechanism to convert the meta-comment cluster from a persistent loss into a win.

## Caveats (do not overclaim)
- The margin is still small: **+2/818 = +0.0025** over primary_only, **+1 sample over C**.
  The temp-0 capture drifted to net +3 (final 0.7857) vs the headline net +2 — i.e. ±1-sample
  noise persists. The *mechanism* (fixing 4 meta-comment breaks, costing 2 rescues) is
  principled and repeatable, but the exact net is noise-adjacent.
- **The gate can hurt when the primary is wrong-neutral on genuinely evaluative text** (the 2
  hurt cases). On a *weaker* primary (C3 generated), the primary is wrong-neutral far more
  often, so a primary-protecting gate could cost more rescues there — **this is the key thing
  the C3 check must verify.** Do not promote to default before that.
- Per-agent rows are from the instrumentation re-run (≤1-sample drift), not the official
  headline transition counts.

## Recommendation / next
- **Lead sentiment config = G** (C trio + non-voting IntentGate). Keep C as the simpler
  fallback. Retire D/F; keep B/E opt-in only.
- **Next (not run here): validate G on the C3 generated-primary** — specifically to check the
  gate does not erode the +0.059 weak-primary gain by blocking rescues of a frequently-wrong
  neutral primary. This is the decisive open question for G.

## Artifacts
- Headline: `experiment_ahmed_designG_intent_gate/full/` (metrics, predictions, llm_usage, run.log).
- Capture: `…/error_attribution/attribution_table.{csv,json}` (per-agent + `gate_blocked` /
  `pre_gate_label`); driver `scripts/ahmed_designG_attribution.py`.
- Guard implementation: `src/agents/consensus_agent.py` (`intent_gate`), wiring in
  `evaluate_pipeline.py`; design rationale in `EXPERIMENT_AHMED_PROMPT_CLUES_AGENT_MAPPING.md`.
