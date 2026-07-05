# Why Did Stage 3 (Pragmatic Verifier) Revise Only 3/84? — Analysis

**Analysis only — no code, no runs, no LLM calls.** Reasons about the observed
inertness of the sequential_sentiment_v1 pragmatic stage (81 keep / 3 revise on the
Ahmed escalated subset), grounded in the actual Stage-2 and Stage-3 prompt text and the
captured decision trace. Date: 2026-07-01.

## The observation
On 84 escalated samples the controller decided: **58 `polarity_kept`, 23
`intent_no_opinion`, 3 `pragmatic_revision`, 0 fallback.** Stage 3 chose **keep 81 /
revise 3** — it changed the answer ~3.6% of the time. The question is whether that is a
*finding about the task* or an *artifact of how Stage 3 was built*. The honest answer is:
**mostly the latter, with a genuine-scarcity floor underneath.**

---

## 1. Is Stage 3 an independent reasoner, or a validator of Stage 2? — **A validator.**

By construction it is a **validator, not an independent reasoner.** Three structural tells:

- **It is handed the answer.** Stage 3's user prompt embeds the full Polarity JSON
  (`label`, `confidence`, `reasoning`, `evidence`). It does not compute a polarity from
  scratch; it is asked to *ratify or overturn a specific proposed label.*
- **Its output space is keep/revise, not classify.** The task is framed as a binary
  review (`keep_or_revise`), and on `keep` the `final_label` is *required* to equal the
  incoming label. So the "do nothing" branch is the structurally default, lowest-effort
  action.
- **Its mandate is explicitly restrictive.** The system prompt says *"Revise ONLY when
  you have a specific pragmatic reason (sarcasm, implicature, or clear
  description/mention). Otherwise KEEP the proposed polarity."* That is an instruction to
  keep by default and revise only on a narrow trigger set.

A true independent reasoner would be asked "what is the polarity?" and its disagreement
with Stage 2 would then be *measured*. Ours is asked "is this polarity OK?" — a different,
weaker question that presupposes the Stage-2 answer.

## 2. Does the Polarity JSON anchor GPT toward confirmation? — **Yes, strongly.**

Providing the proposed label plus its *reasoning and evidence* is a textbook anchoring /
sycophancy setup. Two compounding mechanisms:

- **Anchoring:** once a plausible label with a justification is in context, the model
  treats it as the prior to be defended. Overturning it requires the model to contradict a
  stated, evidenced conclusion — a higher bar than forming a fresh judgment.
- **Sycophancy/agreement bias:** instruction-tuned models are trained to be agreeable and
  to avoid gratuitous contradiction; "the previous step already reasoned X and gave
  evidence" reads as a soft cue to concur unless there is glaring conflict.

So the 81/3 split is *exactly* the direction anchoring predicts. The stage is not
independently sampling the pragmatic reading of the text; it is grading Stage 2's paper
with Stage 2's own answer key visible.

## 3. Would text + Intent only (Stage 3 forms its own polarity) be more independent? — **Yes, but it re-creates the parallel design and does not escape the ceiling.**

If Stage 3 received only the text + Intent JSON (not the Polarity JSON) and produced its
*own* polarity, then a disagreement would reflect two genuinely independent reads, and the
"revision" rate would rise well above 3/84. **That is a real fix for the *independence*
problem — but it has two consequences we already have evidence about:**

- It converts Stage 3 from a *verifier* into a **second independent polarity estimator**,
  i.e. the pipeline becomes a 2-model ensemble (Stage 2 vote + Stage 3 vote) plus the
  intent gate. **That is structurally the parallel design we already tested** (Polarity +
  Contextual + IntentGate), which is *at ceiling* on the strong primary.
- The two reads share **one base model on one text**, so their errors are **correlated**
  (the parallel analysis measured ~81% agreement and lone-dissent precision ≤0.58). More
  disagreement ≠ more accuracy; an independent Stage 3 would surface more flips, but the
  net would land back where parallel voting already lands.

**So independence raises the *action rate* but not the *ceiling*.** The useful reframing:
the reason to make Stage 3 independent is not to beat the primary (it won't) — it is that a
*verifier* over a same-model prediction is close to a no-op, so if Stage 3 exists at all it
should be an independent estimator, not a reviewer. But then it is no longer "sequential
reasoning" — it is parallel voting with extra steps.

## 4. Vs Ahmed's prompting — did Ahmed review a prior prediction? — **No. Each prompt independently solved a sub-task.**

Ahmed's described strategy is a **forward decomposition**: evidence → opinion/intent
existence → target → polarity → decision. Each prompt **solves a fresh sub-question and
adds information**; there is **no step that hands the model a previous *prediction* and
asks it to critique that prediction.** The chain accumulates evidence, it does not
self-review.

Our Stage 3 is a **self-review / verify** step — a pattern **Ahmed did not use.** This is
the precise point where sequential_sentiment_v1 *diverges* from Ahmed:

- **Ahmed (forward-decomposition):** each stage's output is *new evidence* consumed by the
  next stage's *independent* decision. No stage re-judges an earlier label.
- **Ours (Stage 3):** a stage re-judges an earlier *label*. Handing back a prediction for
  ratification is the confirmation-prone move, and it is exactly the move Ahmed avoided.

Stages 1→2 of ours *do* follow Ahmed's forward pattern (intent is new information feeding
an independent polarity). **Stage 3 is the un-Ahmed-like addition, and it is the inert
one.** That is a coherent, not coincidental, result.

## 5. Is Stage 3 structurally redundant with Stage 2? — **Largely yes.**

Read the Stage-2 prompt: it already instructs the model to handle *"explicit praise/insult,
and IMPLICIT praise/insult (sarcasm-free implication, admiration, or put-downs with no
explicit sentiment word),"* negation, intensifiers, and mixed polarity, and to *"judge the
author's stance, not the sentiment of a quoted/mentioned/described thing."* **Stage 2
already performs most of the pragmatic reasoning.** It is explicitly told to *defer only
sarcasm/irony* to Stage 3.

So Stage 3's **unique, non-redundant slice is just sarcasm/irony** — plus a
description-vs-evaluation check that **also overlaps Stage 1's `use_vs_mention`.** Its
genuinely exclusive territory is therefore a thin sliver (literal-meaning inversion), and
even that is partly covered upstream. Structurally, given how Stages 1 and 2 are worded,
**Stage 3 is close to redundant** — it can only add value on the rare cases where the true
stance *inverts* the literal reading and neither Stage 1 (mention) nor Stage 2 (implicit)
already caught it.

## 6. Attribution of the 3/84 revision rate (reasoned estimate — not measured)

The low rate is **over-determined**; several causes push the same direction. A rough
decomposition (estimate, would need an A/B to confirm — see caveat):

| cause | est. share | reasoning |
|---|---|---|
| **Prompt wording** ("revise ONLY…otherwise KEEP") | **~35%** | Explicitly instructs keep-by-default and restricts revision to a narrow trigger set. |
| **Confirmation / anchoring** (Polarity JSON shown) | **~25%** | Presenting an evidenced answer biases toward ratification; sycophancy. Coupled to wording. |
| **Architecture / redundancy** (Stage 2 already did the pragmatic work; Stage 1 covers mention) | **~25%** | Little is *left* for Stage 3 to find; its exclusive slice is thin and partly upstream. |
| **Genuine lack of additional information** (true sarcasm-flips are rare in 84) | **~15%** | Even an ideal independent reviewer would flip only a handful — sarcasm/irony is genuinely uncommon, and the parallel Contextual "sarcasm edge" was already weak and inconsistent (lone-dissent precision 0.58, fails inside its own domain). |

**Reading:** the *dominant* causes are **removable** (wording + anchoring ≈ 60%) — a
differently-built Stage 3 (independent, or with a permissive/critique framing) would revise
more often. But the *floor* is **not** removable: redundancy + genuine scarcity (≈40%)
means even a perfectly independent Stage 3 would still intervene on only a modest number of
cases, and — per Q3 and all prior ceiling evidence — those extra interventions would be
correlated and net-neutral on the strong primary.

---

## Synthesis — the more fundamental point

**A verify/review stage placed over a same-model prediction collapses toward confirmation;
it is structurally close to a no-op.** That is the real finding, and it is consistent with
Ahmed's design *avoiding* self-review in favor of forward decomposition. Three takeaways:

1. **Stage 3 as built is a validator, and validators of your own model's output are
   near-inert by anchoring** — the 3/84 is what that pattern produces, not a property of
   the data alone.
2. **Most of the "pragmatic reasoning" already happens in Stage 2**, which was written to
   absorb implicit praise/insult and mention-vs-use; Stage 3's exclusive contribution
   (sarcasm inversion) is a thin, rare, and partly-upstream slice → **largely redundant.**
3. **Making Stage 3 independent would raise its action rate but re-creates parallel voting
   and re-inherits the correlation ceiling** — so it changes the *shape*, not the *bound*.
   The lever with headroom remains the **weak primary**, not the internal reasoning
   topology.

## Caveats
- The 3/84, 58/23/3 split, and 0-coercion faithfulness are **measured** (decision-trace
  capture). The **attribution in §6 is a reasoned estimate**, not a measurement:
  cleanly separating "wording" vs "anchoring" vs "redundancy" would require a small A/B
  (e.g. independent-Stage-3 vs verifier-Stage-3, and a permissive-wording vs
  restrictive-wording variant) — **not run here, per the no-paid-run constraint.**
- All numbers are a single temp-0 draw (±1–2 sample noise).
- Cross-validated against `EXPERIMENT_AGENT_SPECIALIZATION_ANALYSIS.md` (Contextual's weak,
  inconsistent sarcasm edge) and `EXPERIMENT_CONSENSUS_INVESTIGATION_SUMMARY.md` (the
  express-vs-mention gate is the only lever that moved) — both consistent with a nearly-
  inert pragmatic layer.

## Artifacts
- Trace: `experiment_seqv1_ahmed/decision_trace/trace_table.{json,csv}` (decided_by,
  keep/revise, coercion flags).
- Prompts analyzed: `src/prompts/sequential_sentiment_prompts.py`
  (`POLARITY_SYSTEM_PROMPT`, `PRAGMATIC_SYSTEM_PROMPT`).
- Related: `EXPERIMENT_SEQUENTIAL_SENTIMENT_V1_AHMED_RESULTS.md`,
  `EXPERIMENT_SEQUENTIAL_SENTIMENT_AGENT_DESIGN.md`.
