# Pragmatic Reasoner — Prompt Design & Refine-vs-Replace Analysis (design only)

A single **Pragmatic Reasoner** prompt that explicitly decomposes the remaining pragmatic
tasks (speech act · target attribution · mention-vs-use · implicature · description-vs-
evaluation) *before* assigning sentiment, keeping the architecture unchanged. Includes a
conceptual comparison against **IntentGate** and **Contextual**, and a verdict on whether
it should **replace** or **refine** Contextual. **Design only — no implementation, no runs,
no LLM calls.** General sentiment reasoning; no dataset/benchmark named. Date: 2026-07-01.

## Motivation (why pragmatics, why now)
The reachable ceiling on the strong (Ahmed) frozen primary is a **pragmatic-reasoning
ceiling, not a sentiment-reasoning one**: of the ~19 escalated errors, ≈18 are pragmatic
(implicature, sarcasm/irony, mention-vs-use, target attribution, speech-act recognition,
description-vs-evaluation) and only ≈1 is a genuine sentiment/lexical miss. Sentiment
*mapping* (cue → polarity) is essentially solved; the residual is the pragmatic *prerequisite*
to that mapping. (See `EXPERIMENT_G_TO_093_GAP_ANALYSIS.md` and the pragmatic-vs-sentiment
analysis.) The Pragmatic Reasoner targets exactly that layer.

---

## 1. The Pragmatic Reasoner prompt (proposed)
Fits the unchanged framework: identical JSON contract (`label / confidence / reasoning /
evidence`), occupies the **existing Contextual voting slot** (one vote — NOT a 4th agent),
general wording (no benchmark names; annotation conventions reframed as pragmatic
inferences rather than hard rules). The 5-step decomposition is **internal**; only a
one-sentence rationale surfaces, so the parser and token budget are unchanged.

```
You are a PRAGMATIC REASONING specialist in a multi-agent text classification system.
Before assigning any sentiment, you first resolve the pragmatic structure of the message,
then decide sentiment from that structure. You reason about the whole message and its
communicative intent — not isolated words.

REASONING ORDER — work through these five questions in order, then decide:

1. SPEECH ACT — what is the author DOING? (stating an opinion, asking a question, giving
   advice/a request, promoting/advertising, quoting/reporting, greeting, joking, or
   describing content). Non-evaluative acts (questions, requests, promotions, quotes,
   pure descriptions) often carry NO evaluation of the author's own.

2. TARGET ATTRIBUTION — whose attitude, toward what? Separate the AUTHOR'S own evaluation
   from the author reporting, asking about, or reacting to OTHER people's actions,
   reactions, or opinions. An opinion about what others did is not the author's own stance.

3. MENTION vs USE — is a sentiment-bearing or platform term (e.g. like/dislike/unlike,
   comment, share, a named work/brand) being USED to express the author's evaluation, or
   merely MENTIONED, referenced, or counted? A referenced token is not an expressed opinion.

4. IMPLICATURE — is a stance IMPLIED rather than stated? Detect implicit insult, mockery,
   sarcasm/irony (surface polarity may INVERT), veiled or backhanded praise, and rhetorical
   questions that carry a stance. Do not require an explicit sentiment word.

5. DESCRIPTION vs EVALUATION — is the author recounting events, plot, or content, or
   evaluating them? Narrated or quoted content is not the author's evaluation.

DECISION — after resolving 1–5:
- If the author expresses an evaluation, output its polarity (positive/negative),
  applying any irony/sarcasm inversion from step 4.
- If no author evaluation is expressed (a non-evaluative speech act, a mention/reference,
  or a description/report of others), output neutral or lower confidence.
- Calibrate confidence to how clearly the pragmatic structure supports the decision;
  prefer neutral/low confidence when the message is too short or ambiguous to infer a stance.

RULES — follow every rule exactly:
A. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
B. Decide from the pragmatic structure above — not from surface words or emojis alone.
C. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
D. The JSON must contain exactly these four keys:
   - "label"      : string — one of the allowed labels, copied verbatim
   - "confidence" : float  — 0.0–1.0
   - "reasoning"  : string — one sentence naming the decisive pragmatic factor (speech act /
                    target / mention-vs-use / implicature / description-vs-evaluation) and the label
   - "evidence"   : array  — 1–5 short phrases from the text

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```

**How it drops in without architecture change:** it replaces the *prompt* of the existing
Contextual agent (same agent, same `contextual_output` slot, same single vote). No new
slot, no consensus change, no router change. Would be gated as a prompt/agent variant
(e.g. a `pragmatic` Contextual prompt), default unchanged.

---

## 2. Conceptual comparison — Pragmatic Reasoner vs IntentGate vs Contextual
| dimension | Contextual (current) | IntentGate | Pragmatic Reasoner (proposed) |
|---|---|---|---|
| **Role** | holistic whole-message interpreter (sarcasm, tone, implicit) | one pragmatic question: "opinion expressed, or meta/mention?" | **all five** pragmatic subtasks → then sentiment |
| **Scope** | broad but **unstructured** | narrow (1 axis) | broad **and structured** (explicit 5-task decomposition) |
| **Pipeline position** | a **voter** (contextual slot) | **non-voting guard** (pre-consensus veto) | a **voter** (contextual slot) |
| **Effect on consensus** | 1 of 3 votes; can be outvoted by Lexical+Polarity | vetoes an unsupported override; protects the primary | 1 of 3 votes; a **sharper** pragmatic vote |
| **Pragmatic tasks covered** | implicature/sarcasm (implicitly), some target | speech-act + mention-vs-use only | speech-act, target, mention-vs-use, implicature, description-vs-evaluation |
| **Failure mode** | vague → sometimes agrees with the surface bloc (still the strongest agent, ~0.75) | over-blocks affective meta-questions (G2); can't recover fan cheer | over-decomposition in one call; risk of over-neutralizing if the speech-act/mention steps dominate (F-like) |
| **Relation to the others** | the role being upgraded | a *subset* of the Reasoner's tasks, reused as a veto | superset of IntentGate's axes; structured form of Contextual |

**Two structural facts:**
- **Vote vs veto.** IntentGate is a *non-overriding guard*; Contextual/Pragmatic-Reasoner is a
  *vote*. They act at different points and are **complementary, not substitutes** — a veto
  protects against a wrong bloc; a better vote improves the bloc. The Reasoner does **not**
  make the gate redundant: a single pragmatic voter can still be outvoted by Lexical+Polarity
  over-reading surface cues, so the gate's independence remains valuable.
- **Same role, made explicit.** The Pragmatic Reasoner *is* Contextual's function (whole-message
  pragmatic interpretation), decomposed and made exhaustive. IntentGate's two axes (speech-act,
  mention-vs-use) are a proper subset of the Reasoner's five.

---

## 3. Verdict — **REFINE Contextual (in-place prompt upgrade); do NOT replace it and do NOT add a 4th agent**
Three reasons, each tied to prior evidence:

1. **It is the same role, not a new one.** "Replace" implies discarding Contextual's function;
   the Pragmatic Reasoner is that function, decomposed. The correct operation is an **in-place
   prompt upgrade of the existing Contextual agent** — same agent, same slot, same single vote.
   Architecture is literally unchanged.
2. **Adding a pragmatic voter as a 4th agent is already known to be a wash.** Design D (4 voters)
   and Design E (Intent-as-4th-vote) both tied C at higher cost. A new pragmatic *voter* beside
   Contextual would just be another correlated bloc member. Refining the *existing* Contextual
   vote preserves the 3-vote structure (Lexical = cues · Polarity = decision · Contextual =
   pragmatics) and its decorrelation.
3. **Contextual is the strongest and most independent agent (≈0.75 in every design).** Upgrading
   a working, load-bearing component's *reasoning* is lower-risk than swapping in a new agent,
   and it targets the ceiling (pragmatic inference) at the exact slot that already does
   pragmatics best.

**Keep IntentGate unchanged** (the non-voting guard). The combined design becomes:
**Lexical (cues) + Polarity (decision) + Pragmatic-Reasoner Contextual (structured pragmatic
vote) + IntentGate (pragmatic veto)** — vote and veto attacking the pragmatic ceiling from two
sides. Do not fold Polarity into the Reasoner (that would re-correlate the decision and the
pragmatic vote); keep separation of concerns.

---

## 4. Expected effect on the 19 reachable errors (conceptual)
- **Plausibly helped (pragmatic-inference cases):** implicature/sarcasm (`00097`, `00046`),
  description-vs-evaluation (`00193`), some target/mention (`00220`) — where an *explicit*
  pragmatic step could flip Contextual's vote off the surface read. Maybe **2–4**, and only if
  the improved vote then survives consensus (Lexical+Polarity may still over-read).
- **Not helped:** cue-less implicit insults/praise (`00008`, `00182`, `00635`, `00642`) —
  decomposition creates no signal that isn't inferable from terse bilingual text (these needed
  *fine-tuning* in Ahmed's work); and label-convention cases (`00295`, `00250`) unless
  conventions are hard-coded (dataset-tailoring), which the prompt deliberately avoids.
- **New risk:** the speech-act + mention steps could push Contextual to **over-neutralize** (an
  F-like failure), degrading the strongest agent into a neutral-leaner. Mitigations built into
  the prompt: the "if the author's own stance is present, use it" clause and explicit confidence
  calibration.

## 5. Risks & mitigations (for the eventual A/B)
| risk | mitigation |
|---|---|
| Over-neutralization (speech-act/mention steps dominate → F-like) | the stance-override clause + confidence calibration; keep it a *vote* (Lexical/Polarity balance it) |
| Correlation with Polarity (both "decide") | Reasoner reasons about *structure/intent*; Polarity decides *polarity of an expressed evaluation* — keep the split explicit |
| Single-call over-decomposition (5 steps done shallowly) | steps are ordered and few; only a 1-sentence rationale surfaces (no CoT bloat) |
| Convention cases still wrong | out of scope by design (general, not dataset-tailored) |

## 6. Honesty / expected ceiling
- A refined Pragmatic-Reasoner Contextual is the **right next lever** — it targets the actual
  bottleneck (pragmatics) at the right slot with the architecture unchanged. But its realistic
  gain is **~1–4 samples, noise-adjacent**, because ~⅓ of the residual is *convention* (not
  reasoning) and the hardest implicit cases are beyond single-pass inference.
- It is more likely to **sharpen vote quality and decorrelate the panel** than to *robustly*
  clear 0.930 on the strong primary. Consistent with the gap analysis, a comfortable >0.93 still
  needs a stronger/better-calibrated primary (retraining) or convention encoding (tailoring).
- **Where it may matter more:** on a *weaker* primary (C3 generated), the panel does more of the
  work and better pragmatic reasoning has more headroom — so the Pragmatic Reasoner is worth
  A/B-ing on **both** primaries, and is a natural companion to the pending C3 check.

## Recommended (not yet approved) next step
Implement the Pragmatic Reasoner **as a refinement of the Contextual prompt** (opt-in variant,
default + G unchanged), keep IntentGate as the guard, and A/B it on the Ahmed frozen primary
(and later C3). Do **not** add it as a 4th agent. **Design only — nothing implemented or run.**

## Artifacts / basis
- `EXPERIMENT_G_TO_093_GAP_ANALYSIS.md` (the 19 reachable errors, typed).
- `EXPERIMENT_SENTIMENT_INTENT_GATE_ABLATION.md` (G), `EXPERIMENT_SENTIMENT_SELECTIVE_INTENT_GATE_G2.md` (gate limits).
- `EXPERIMENT_AHMED_PROMPT_CLUES_AGENT_MAPPING.md` (Ahmed's pragmatic/convention layer).
- Current Contextual prompt: `src/prompts/contextual_prompt.py` (semantic_v1 addendum).
