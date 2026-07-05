# Design G — Prompt Instructions vs Actual Failures

**Analysis only — no LLM calls.** Maps specific Design-G failure cases (Ahmed escalated)
onto the exact prompt instruction that *should* have caught each, to see whether the
failures are (a) **prompt-coverage** gaps, (b) **model-compliance** failures (the rule is
present, the model ignored it), (c) **knowledge** gaps, or (d) **architecture**
limitations. Uses the saved attribution table + the four agent prompts. Date: 2026-07-02.

## The key result up front
**Every recurring failure is already covered by an explicit prompt instruction.** The
prompts do *not* lack the right rule — the model fails to comply with it, lacks the cultural
knowledge to apply it, or the gate architecture can't act on it. This is why every
prompt-wording redesign (semantic_v1 → v3 → sequential v2) plateaued or hurt: **the bottleneck
is not prompt coverage.**

## Case-by-case mapping

### Case 00240 — "على اساس ايه اللى عاملين dislike" (true **neutral** → G **negative**)
Votes: lex=neg, pol=neg, ctx=neg, **gate=neutral** (correct!), primary=neg.
| agent | instruction that should have fired | what happened |
|---|---|---|
| Lexical | *"Treat platform words (like, **dislike**, unlike, comment…) as WEAK cues unless the author states their own opinion"* | **Ignored** — voted negative on the word "dislike". |
| Polarity | step 3: *"if the text only MENTIONS platform actions… choose neutral or low confidence"* | **Ignored** — voted negative. |
| Contextual | *"Decide whether the text is a… platform interaction… do NOT overrule a neutral reading just because emotional words appear"* | **Ignored** — voted negative. |
| IntentGate | Q4: *"platform/meta-comment about likes, **dislikes**… → neutral"* | **Followed** — said neutral. But **architecturally powerless** (see below). |
**Verdict:** prompt-coverage is perfect (all four name "dislike" explicitly); this is a
**model-compliance** failure by 3 agents, and an **architecture** failure for the 1 that got
it right.

### Case 00446 — "Dislikes كتير اوي" (true **neutral** → G **negative**)
Votes: lex=neg, pol=neg, ctx=neg, **gate=neg**, primary=neg — *everyone* wrong, including the gate.
**Verdict:** pure **model-compliance/knowledge** failure. The same "dislike = weak/meta"
rule is in all four prompts; on this ultra-short meta comment the model reads the button
name as sentiment across the board. No prompt wording would fix what the model won't apply.

### Case 00021 — "مخرج بيدور ع ممثلة تعمل دور الممحونة … مى عمر" (true **negative** → G **neutral**)
Votes: lex=neg, pol=neg, ctx=neg (**all correct!**), **gate=neutral → BLOCKED** → final neutral.
- The three voters correctly followed *"detect implicit insult / mockery"* and all said negative.
- The **IntentGate misread** it as no-stance/meta (violating its own Q5 *"is the stance
  implicit / sarcastic?"*) and, because gate==primary(neutral) while winner=negative, the
  veto **overrode three correct agents.**
**Verdict:** **architecture + gate-knowledge** failure — the one case type where the gate's
veto actively destroys a correct answer. (Also 00706: lex=pos, ctx=pos correct, gate vetoed → neutral.)

### Case 00008 — "هوة معتز مسعود دة gay ?" (true **negative** → G **neutral**)
Votes: lex=neu, pol=neu, ctx=neu, gate=neu, primary=neg(0.67).
- Contextual's *"detect implicit insult"* and Polarity's *"implicit"* clause were **in force**
  but the model does not read the slur-as-question as an insult.
**Verdict:** pure **knowledge/pragmatic floor** — cultural reading the base model lacks. No
instruction gap.

## The architectural finding (concrete and important)
The IntentGate can **only protect a neutral primary** from a polar override:
`if winner != primary and gate == primary → revert to primary`. So:
- On **00240 / 00298** the gate *correctly* said neutral, but the primary was already
  **negative** and the agents agreed — there was no override to block, so the gate's correct
  read was **unused**. The gate is powerless to pull a *wrong-polar* primary toward neutral.
- On **00021 / 00706** the gate *wrongly* said neutral and the primary was neutral, so it
  **did** act — and vetoed correct agents.

**Net:** the gate helps only in one direction (protecting a neutral primary) and its two
Ahmed interventions on failures were both *harmful*. Its correct reads (on wrong-polar
primaries) are architecturally inert. This asymmetry is why the gate nets only ~+1–2 and why
loosening it (to let it create neutrals) would re-introduce over-neutralization on the
implicit-stance cases.

## Failure-type tally (the 18 escalated errors)
| type | count | fixable by prompt wording? |
|---|---|---|
| Model-compliance (rule present, ignored) — mostly "dislike"/meta | ~5 | **No** (rule already explicit; v2 showed *more* authority hurts) |
| Knowledge / pragmatic floor (implicit Egyptian insult/praise) | ~7 | **No** (needs stronger/knowledge-augmented model) |
| Gate architecture (correct read powerless, or wrong veto) | ~2–4 | Partly (architecture, not wording) — but trade-off negative |
| Emoji/rhetorical over-read of true-neutral | ~2 | **No** (rule "don't over-read emojis" already present, ignored) |

## Conclusion
**The prompts already encode exactly the right rules** — express-vs-mention, "dislike =
weak/meta", detect implicit stance, don't over-read emojis. The residual failures are the
model **not complying** with rules it was given (the "dislike" cases), **lacking the cultural
knowledge** to apply them (implicit Egyptian insults), or the **gate's one-directional
architecture**. None of these is a wording problem, which is the direct explanation for why
semantic_v1/v3 and sequential v1/v2 could not move the strong-primary ceiling: **you cannot
prompt-engineer past a compliance/knowledge ceiling.** The levers that remain are a
**stronger/knowledge-augmented base model** or the **weak-primary regime** (C3), where the
errors are recoverable ones the model *can* read.

## Artifacts
- `experiment_ahmed_designG_intent_gate/error_attribution/attribution_table.json`
- `_G_ahmed_failures.txt`, `EXPERIMENT_G_AHMED_FAILURE_ANALYSIS.md`
- Prompts: `src/prompts/{llm_lexical,polarity,contextual,intent}_prompt.py`
- Gate logic: `src/agents/consensus_agent.py` (intent_gate guard).
