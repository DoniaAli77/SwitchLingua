# Sequential (Staged-Reasoning) Sentiment Architecture — Design Only

**Design proposal only. No code, no runs, no LLM calls.** Evaluates whether sentiment should be a
*staged reasoning pipeline* (evidence → intent → polarity → pragmatic verify → deterministic
decision) rather than the current *parallel voters* (Lexical + Polarity + Contextual + non-voting
IntentGate). General sentiment analysis; no dataset named in any prompt. Date: 2026-07-01.

---

## 0. Why this is even worth proposing (the motivating diagnosis)

Three prior findings set up the motivation precisely:

1. **The parallel agents are correlated generalists, not specialists.** They emit identical labels
   on **81% of escalated** cases and ~96% overall (`EXPERIMENT_AGENT_SPECIALIZATION_ANALYSIS.md`).
   Root cause: **each agent predicts the same final 3-way label**, so they re-solve one problem and
   their errors correlate — violating the ensemble premise.
2. **The one aggregation change that ever helped was the IntentGate** — and it helped precisely
   because it is **not a vote**: it is a domain-restricted, non-overriding **veto** on the
   express-vs-mention axis (`EXPERIMENT_CONSENSUS_INVESTIGATION_SUMMARY.md` §6). Same pragmatic
   signal: **12/12 suppressed as a vote (Design E) but 0 missed as a veto (Design G).**
3. **SwitchLingua's quality agents worked because each checked a *different property*** (a
   *conjunctive* pipeline), not because they voted on the same output.

The sequential design generalizes finding (2): instead of one veto bolted onto a vote, make **every
stage output a different intermediate property**, and let a deterministic controller compose them.
This is structurally what makes the gate work and what made SwitchLingua work.

---

## 1. What each stage outputs

Each stage answers a **distinct sub-question** and — crucially — **stages 1–2 never output a final
sentiment label.** Only Stage 3 assigns polarity, and only Stage 4 may revise it.

| stage | question it answers | outputs (NOT the final label) |
|---|---|---|
| **1 Evidence / Lexical extractor** | *What explicit cues are present?* | list of sentiment cues (span + type: praise/insult/emoji/emoticon), negation markers, intensifiers, repetition/elongation, platform/meta words, cue polarity *of each span* (not of the text) |
| **2 Intent / opinion-expression detector** | *Is an evaluative stance expressed, and about what?* | `opinion_expressed` (yes/no/unsure), `target` (entity/topic/none), `speech_act` (evaluate / describe / ask / advise / quote), `use_vs_mention` (use / mention / platform-meta) |
| **3 Polarity resolver** | *Given evidence + intent, what polarity?* (only if opinion exists) | `polarity` (pos/neg/neu), `mixed` flag, which Stage-1 cues it used, negation-applied flag |
| **4 Pragmatic verifier / Contextual checker** | *Does pragmatics overturn the literal read?* | `sarcasm/irony` (yes/no), `implicit_stance` (pos/neg/none), `revision` (keep / flip-to-X) + short reason |
| **5 Final decision (deterministic controller)** | *Compose intermediates — no LLM* | final label + provenance (which stage decided) |

The key difference from parallel: **Stage 3 is *conditioned on* Stage 1 and 2's structured output**,
and **Stage 4 is conditioned on Stage 3.** Information flows forward; it is not independently
re-derived four times.

---

## 2. JSON schemas (illustrative, general — no dataset terms)

```jsonc
// Stage 1 — Evidence extractor
{
  "cues": [
    {"span": "wallah beautiful", "type": "praise", "cue_polarity": "positive"},
    {"span": "block", "type": "platform_meta", "cue_polarity": "neutral"}
  ],
  "negation": true,
  "intensifiers": ["so", "!!!"],
  "elongation_repetition": true,
  "platform_or_meta_terms": ["block", "trending"],
  "notes": ""            // no final sentiment here
}

// Stage 2 — Intent / opinion-expression
{
  "opinion_expressed": "yes",        // yes | no | unsure
  "target": "the movie",             // entity/topic | null
  "speech_act": "evaluate",          // evaluate | describe | ask | advise | quote
  "use_vs_mention": "use",           // use | mention | platform_meta
  "confidence": 0.0                  // reserved; see §11 (serialize it this time)
}

// Stage 3 — Polarity resolver (only invoked if Stage 2 opinion_expressed != "no")
{
  "polarity": "positive",            // positive | negative | neutral
  "mixed": false,
  "used_cues": ["wallah beautiful"],
  "negation_applied": false,
  "confidence": 0.0
}

// Stage 4 — Pragmatic verifier
{
  "sarcasm_irony": false,
  "implicit_stance": "none",         // positive | negative | none
  "revision": "keep",                // keep | flip_positive | flip_negative | flip_neutral
  "reason": "literal praise, no ironic markers",
  "confidence": 0.0
}

// Stage 5 — Deterministic controller output (no LLM)
{
  "final_label": "positive",
  "decided_by": "stage3_polarity",   // provenance for audit
  "gate_applied": false
}
```

### Stage-5 controller logic (deterministic, no voting)
```
if stage2.opinion_expressed == "no"        # or use_vs_mention in {mention, platform_meta}
   and stage4.implicit_stance == "none":
        final = neutral                     # <-- this is the IntentGate, now first-class
elif stage4.revision != "keep":
        final = stage4.revision             # pragmatic override (sarcasm / implicit stance)
elif stage2.opinion_expressed in {"yes","unsure"} and stage1.cues non-empty:
        final = stage3.polarity             # trust resolver when evidence + intent align
else:
        final = stage3.polarity or neutral  # fallback
```
Note this controller is **the union of the current gate + a resolver-trust rule**, made explicit and
ordered, rather than emergent from a weighted vote.

---

## 3. Which current errors the sequential design could fix

Grounded in the 16 real disagreement cases (`EXPERIMENT_AGENT_SPECIALIZATION_ANALYSIS.md`):

- **Mention-vs-use / platform-meta neutrals** (00363, 00449, 00239, 00396 — meta-questions, brand
  mentions → true `neutral`). Stage 2 sets `use_vs_mention = mention/platform_meta`,
  `opinion_expressed = no`; Stage 5 forces neutral **before** any polar cue is scored. The parallel
  panel gets these only *sometimes* (via the gate); the pipeline makes it the **default path**, so it
  should be more consistent — and it removes the 23/84 run-to-run gate-label instability by making
  the neutral decision a structured branch rather than a fragile veto label.
- **"Author's own stance" false-neutrals** (00487, 00517, 00706 — author expresses a positive act,
  true `pos`, Polarity over-neutralized). Stage 2 = `use`, `speech_act = evaluate`,
  `opinion_expressed = yes` → Stage 5 **trusts Stage 3** instead of letting a neutralizing bloc
  outvote the real cue. This is exactly the class the parallel Polarity/Contextual camp
  *over-neutralizes*.
- **Implicit insult / sarcasm** (00100, 00182 — no explicit cue, true `neg`). Stage 4 runs on the
  *resolved* polarity and can flip it, and its verdict is composed deterministically rather than
  being one vote among a correlated bloc that outvotes it.
- **Structural consensus loss** generally: the oracle showed **4–11 correct answers/84 are held by
  some agent but discarded** by the vote (`…INVESTIGATION_SUMMARY.md` §2). A pipeline that **routes
  by intermediate property** cannot "outvote" the stage that owns the decision — it removes the
  mechanism that produces consensus loss.

## 4. Which errors would remain (be honest)

- **The ~⅔ unrecoverable floor.** Of ~19–23 escalated errors per design, **~⅔ had the truth in
  *no* agent** (`…INVESTIGATION_SUMMARY.md` §2). Sequential reasoning re-runs the *same base model*
  (gpt-4o-mini) on the same ambiguous code-switched text; if no lens recovers the label today, an
  ordered pipeline of the same model likely won't either. **H(Y|X) > 0** floor is untouched.
- **Correlated *base-model* errors.** The stages still share **one** model. If that model
  systematically misreads a dialectal cue, every stage inherits the misread — Stage 3 trusts Stage
  1's wrong cue, Stage 4 rationalizes it. Decorrelation needs *heterogeneous models*, which this
  design does **not** provide.
- **Error propagation / cascade.** New failure mode: a wrong Stage 2 (`opinion_expressed = no` on a
  real opinion) **hard-suppresses** Stage 3 → guaranteed neutral. Parallel voting is *robust* to one
  bad agent; a pipeline is *brittle* to an early-stage error. This is a real risk that could **add**
  errors on the "subtle stance present" cases if Stage 2 is imperfect (and Stage 2 is doing the hard
  pragmatic call).
- **The strong-primary ceiling.** On the Ahmed frozen primary the whole agentic layer is at ceiling:
  McNemar shows **no design significantly beats primary_only (p=0.82)**. A better *aggregation shape*
  cannot manufacture signal that isn't in the escalated subset. Expect **~0 significant movement on
  the strong primary** regardless of architecture.

## 5. Would this reduce agent correlation? — **Yes, structurally.**

Correlation today is high **because every agent outputs the same 3-way label** (81% identical).
Under the pipeline:
- Stage 1 outputs **cue spans**, Stage 2 outputs **intent/target/use-mention**, Stage 3 outputs
  **polarity**, Stage 4 outputs **a revision decision** — **different label spaces**, so "agreement
  on the final label" is no longer even defined for stages 1–2. Correlation of *the thing being
  combined* drops by construction.
- The composition is **conjunctive** (each stage must pass its check), like SwitchLingua's working
  quality agents, rather than **disjunctive/averaged** (the vote), which is where correlation hurts.

**Caveat:** this reduces *output* correlation, not necessarily *underlying-judgment* correlation —
all stages still share one model's world-view (§4). So correlation drops in the aggregation sense
but not to the level a heterogeneous-model panel would achieve.

## 6. Cost / latency

- **Latency: worse.** Parallel agents can be issued concurrently (~1 LLM round-trip on the critical
  path). The pipeline is **strictly sequential** — Stage *k* needs Stage *k−1*'s output — so it is
  **4–5 round-trips in series** on the latency path. Roughly **3–5× wall-clock latency** per
  escalated sample.
- **Token cost: comparable-to-slightly-higher.** Similar number of LLM calls (4–5 vs 3–4 today), but
  each later stage's prompt now **includes upstream structured JSON**, so input tokens grow per
  stage. Estimate **+20–40% tokens** vs the current parallel set. Only escalated samples (~10%) pay
  this, so **absolute** cost impact is small.
- **Mitigation:** Stage 1 (evidence) and Stage 2 (intent) are independent and **could run in
  parallel**; only 3→4→5 must be serial. That cuts the latency penalty to ~3 serial hops.

## 7. Offline-testable, or new LLM calls required? — **Requires new LLM calls. Cannot be simulated
offline.**

This is the decisive practical constraint. The saved captures store **per-agent labels from the
*parallel* run** (`lexical_label / polarity_label / contextual_label / gate_label`), i.e. each agent
saw only the *raw text*. The sequential design changes **what each stage sees**: Stage 3 is
conditioned on Stage 1+2's structured output, Stage 4 on Stage 3. **We have no record of what a stage
would output given upstream structured input**, so it cannot be reconstructed from the label tables.

- The offline rescoring/weight-sweep work *could* reuse saved labels because it only **re-fused the
  same parallel outputs**. Sequential is a **different computation graph** → **new paid capture
  required**.
- Only **Stage 5 (the controller)** is offline-testable — but only *given* Stages 1–4 outputs, which
  we don't have. So even the deterministic part can't be validated without first capturing 1–4.

## 8. General / task-aware without dataset tailoring? — **Yes.**

Every stage asks a **generic** sentiment-analysis question (cues, opinion existence, target,
polarity, sarcasm) that applies to any sentiment corpus. No prompt references a dataset, platform, or
benchmark. `use_vs_mention` and `speech_act` are standard pragmatics, not dataset artifacts. It is
**task-aware** (it knows it is doing sentiment) without being **corpus-tailored** — the same
constraint the whole design line has honored.

## 9. Closer to Ahmed's prompting strategy? — **Yes, materially.**

Ahmed's rule-based prompting is described as sequential: **evidence → intent/opinion existence →
target → polarity → final decision.** The proposed Stages 1→2→3→5 are a near-isomorphic mapping, with
Stage 4 (pragmatic verifier) added as an explicit sarcasm/implicature check. So this is the **first
design in the line that mirrors Ahmed's *reasoning structure*** rather than only borrowing agent
roles into a vote. That is a genuine conceptual argument for prototyping it — it tests the hypothesis
"the *sequence* is what carries Ahmed's gains," which the parallel designs never isolated.

## 10. Recommendation — **Build a minimal 3-stage prototype, but test it where headroom exists (weak
C3 primary), not on the strong Ahmed primary.**

**Prototype (minimal), not the full 5 stages:**
- Collapse to **Stage 2 (intent/use-mention) → Stage 3 (polarity) → Stage 4 (pragmatic verify) →
  deterministic controller.** Fold Stage 1 evidence *into* Stage 3's prompt initially (cue
  extraction as a sub-step) to save one hop; split it out only if Stage 3 is weak.
- This isolates the **one hypothesis worth testing**: does *conditioning polarity on an explicit
  opinion-existence/use-mention decision* (a first-class gate) beat *bolting a veto onto a vote*?

**Where to test — this is the load-bearing caveat:**
- **Do NOT expect movement on the strong Ahmed primary.** It is at ceiling (McNemar p=0.82, oracle
  and rescoring all agree). Running the prototype there mainly buys a **cleaner, more stable
  neutral-decision** (removing the 23/84 gate-label instability) — a *robustness* win, not an
  *accuracy* win. Worth reporting but not the point.
- **The decisive test is the weak C3 generated primary**, where the primary term is small and the
  agentic layer carries the decision — the **one regime where any structural improvement can pay
  off** (this is the repeatedly-flagged untested lever).

**Before running anything, add the two data-hygiene fixes we've been blocked by:**
1. **Serialize per-stage confidences** (the `confidence` fields in §2) — every prior confidence-aware
   analysis was impossible because labels-only were stored. Do not repeat that mistake.
2. **Capture on a dev split too**, not just test, so the controller thresholds / stage ordering can be
   tuned without test leakage.

**Bottom line:** The sequential design is the **best-motivated architecture change on the table** —
it directly attacks the correlation root cause, generalizes the one lever (the veto) that ever
worked, and is the closest to Ahmed's actual strategy. But it **cannot be validated offline** (needs
new paid captures), it will **not** significantly move the strong primary (proven ceiling), and it
introduces a **cascade-brittleness** risk. So: **prototype the 3-stage minimal version, instrument
confidences + dev capture, and evaluate it on the weak C3 primary — that single experiment is what
would actually decide whether staged reasoning beats parallel voting.**

## Artifacts / basis (no new computation)
- `EXPERIMENT_AGENT_SPECIALIZATION_ANALYSIS.md` — 81% agreement, the 16 disagreement cases, no
  strong specialization (motivates the redesign).
- `EXPERIMENT_CONSENSUS_INVESTIGATION_SUMMARY.md` §6 — vote-vs-veto (12/12 vs 0), why the gate works.
- `EXPERIMENT_G_OFFLINE_CONSENSUS_RESCORING.md` — label-only re-fusion cannot beat G (why a *new*
  computation graph, not re-fusion, is needed).
- `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md` — the correlation/information floor (the ⅔
  unrecoverable errors that remain).
- Analysis audit — McNemar: no design significantly beats primary_only on the strong primary
  (p=0.82); gate label unstable 23/84 across runs (motivates a first-class deterministic gate).
