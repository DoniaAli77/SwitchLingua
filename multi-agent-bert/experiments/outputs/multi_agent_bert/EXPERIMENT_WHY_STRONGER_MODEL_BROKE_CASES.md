# Why gpt-4.1-mini Broke Predictions 4o-mini Got Right — Per-Agent Diagnosis

**Analysis, with a ~$0.01 per-agent re-capture.** Examines the 5 escalated cases the stronger
model (gpt-4.1-mini, semantic_v1) got wrong that gpt-4o-mini got right, at the individual-agent
level, to find the mechanism. Date: 2026-07-02.

## Per-case agent trace (4o-mini vs 4.1-mini)

| id | true | 4o-mini agents → final | 4.1-mini agents → final | what changed |
|---|---|---|---|---|
| 00041 | neg | lex=neg pol=neg ctx=neg → **neg ✓** | lex=neg **pol=neu ctx=neu** → **neu ✗** | Polarity+Contextual neutralized |
| 00045 | neg | lex=neg pol=neg ctx=neg gate=neg → **neg ✓** | lex=neg pol=neg ctx=neg **gate=neu → BLOCKED** → **neu ✗** | **gate vetoed correct agents** |
| 00100 | neg | pol=neg ctx=neg gate=neg → **neg ✓** | lex=neg pol=neg ctx=neg **gate=neu → BLOCKED** → **neu ✗** | **gate vetoed correct agents** |
| 00127 | neg | lex=neg pol=neg ctx=neg → **neg ✓** | **lex=neu pol=neu** ctx=neg → **neu ✗** | Lexical+Polarity neutralized |
| 00362 | neu | lex=neu pol=neu ctx=neu → **neu ✓** | **lex=neg pol=neg ctx=neg** → **neg ✗** | all agents over-read description |

## Three mechanisms — and they run in OPPOSITE directions

**M1 — The stronger GATE over-vetoes "dislike" as meta (00045, 00100).** The three voters
were **all correctly negative**, but gpt-4.1-mini's IntentGate more aggressively classified the
"dislike"/platform reference as meta→neutral and **blocked the correct negative**. The stronger
model made the *gate* a more aggressive neutralizer, and its one-directional veto turned two
right answers wrong. **This is a gate-architecture × model-strength interaction.**

**M2 — The stronger AGENTS over-neutralize implicit negatives (00041, 00127).** Polarity/
Lexical flipped negative→neutral. The semantic_v1 prompt says *"if the stance is implicit /
platform-related / unclear → prefer neutral or lower confidence."* gpt-4.1-mini **complies with
that guidance more faithfully**; gpt-4o-mini partly ignored it and committed to negative —
which happened to be right. **Better instruction-following = more over-neutralization here.**

**M3 — The stronger AGENTS over-read described content (00362).** Opposite direction: on a
Breaking Bad *plot description* ("فشل جيسي الفاشل"…), 4.1-mini read the negative plot lexicon as
sentiment and all three agents flipped neutral→negative. The stronger model is **more sensitive
to negative vocabulary even inside described content.**

## The unifying insight — 4o-mini was right by ACCIDENT, not understanding
On these genuinely-ambiguous cases, **neither model actually understands** that "123k dislike,
may the message arrive" is the author *endorsing* negativity. The outcome is decided by each
model's **default bias**:
- **gpt-4o-mini's bias:** commit to a polarity (weak neutral-lean compliance) + a less
  aggressive gate → it guesses *negative* and vetoes less.
- **gpt-4.1-mini's bias:** comply with the neutral-lean guidance + a more aggressive meta-gate
  → it guesses *neutral* and vetoes more.

The gold on 4 of these 5 is **negative**, so 4o-mini's "guess negative" bias scored better —
**by luck, not by reading the pragmatics correctly.** The stronger model didn't get *worse* at
understanding; its more-neutral, more-compliant, more-gate-happy behaviour simply lands on the
wrong side of ambiguous cases whose gold happens to be polar. This is the same coin-flip the
"wash" diagnosis described, now confirmed at the agent level.

## Two concrete, actionable consequences

1. **The IntentGate becomes MORE harmful as the model gets stronger.** 2 of the 5 breakages
   (00045, 00100) are the gate vetoing correct agents, because a stronger model detects
   "dislike=meta" more aggressively and the veto is one-directional (it can only neutralize).
   → **Testable prediction: with gpt-4.1-mini, Design G may do BETTER without the gate (plain
   Lexical+Polarity+Contextual consensus) or with the selective gate (G2).** The gate was tuned
   for 4o-mini's weaker meta-detection; it over-fires on a stronger model.

2. **This is exactly why the disambig prompt failed.** It pushed 4.1-mini *harder* toward the
   relationship/neutral reasoning it already over-applies (M1/M2), amplifying the
   over-neutralization instead of fixing it.

## Bottom line
The stronger model didn't lose capability — it **shifted its default bias toward neutral and
toward meta-vetoing**, which loses on ambiguous cases whose gold is polar. 4o-mini's wins there
were **non-compliance that happened to match the gold.** The one *actionable* lever this
exposes is **the gate**: it is now a net liability under a stronger model, so the next cheap
test is **G-without-gate (or G2 selective gate) at gpt-4.1-mini** — not more prompt tuning.

## Artifacts
- Per-agent trace: `experiment_G_ahmed_gpt41mini/broken_why.txt` (script
  `scripts/ahmed_G_broken_why.py`).
- Basis: `EXPERIMENT_G41_WASH_DIAGNOSIS.md`, `EXPERIMENT_G_DISAMBIG_AHMED_RESULTS.md`,
  `EXPERIMENT_G_PROMPT_VS_FAILURE_MAPPING.md` (the gate's one-directional limitation).
