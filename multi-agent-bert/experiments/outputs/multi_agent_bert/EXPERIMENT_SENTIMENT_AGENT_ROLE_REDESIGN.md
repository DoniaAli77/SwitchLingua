# Sentiment Agent Role Redesign — Design Analysis (no implementation)

Prompt-design analysis for the three sentiment specialist agents, grounded in the
completed linguistic error analysis. **Design only — no prompt changes, no runs.**
General reasoning principles only; **not** tailored to any dataset/benchmark. Date: 2026-06-27.

## Evidence basis & a caveat on attribution
- Per-agent outputs are available for one analysed set (the escalated subset with
  captured agent labels). On it: per-agent accuracy **lexical 0.714 · logic 0.679 ·
  contextual 0.726**, and the three agents **agree 92%** of the time (pairwise 93–96%).
- **Because the agents are ~92% correlated, most errors are *shared*, not attributable
  to one agent.** The strongest per-agent signals in the evidence: (a) **Contextual was
  the only agent that ever resisted** the bloc (it stayed correct on a plot-recap and a
  meta-comment where the other two erred); (b) **Logic is the weakest** and mirrors
  Lexical (96% agreement) — i.e. it is **not exercising its distinct structural mandate.**
  Attribution below is stated at that level of confidence; I do not assign error types
  to a single agent where the data shows all three agreed.
- **Out of scope:** the analysis found ~a third of the "agent breaks" were *label-
  convention disagreements where the agent's reading was linguistically defensible*
  (e.g. mildly-positive content scored neutral by the gold label). Those are **not a
  reasoning defect**, and "fixing" them would mean tailoring to a convention — explicitly
  excluded. This redesign targets only the **genuine reasoning errors**.

The genuine, evidence-supported reasoning errors to address:
1. **Surface-cue over-reading** — UI/platform meta-terms (e.g. a "dislike/unlike"
   token), topic/plot nouns, and **emojis** treated as the author's sentiment.
2. **Sentiment-target confusion** — the author describing/asking about *others'* actions
   ("who are the people who disliked this?") read as the author's *own* negativity.
3. **Description-vs-evaluation confusion** — recounting events/plot ("the character
   fails / dissolves the body") read as the author evaluating negatively.
4. **Implicit sentiment missed** — sarcasm, irony, veiled insult or praise that is not
   carried by an explicit lexical cue → defaulted to neutral.

---

## 1. Lexical Agent

### A. Current role
- **Responsibility:** choose a label from **vocabulary cues only** — surface words,
  task-relevant terminology, named entities, salient tokens; cite concrete tokens.
- **Should focus on:** explicit, evaluation-bearing words/phrases actually present.
- **Should ignore:** deep semantics, discourse, intent (by current design).

### B. Observed failure modes (its reasoning)
Its mandate ("surface tokens → label") is the **structural source of surface-cue
over-reading**: it counts the *presence* of a salient token (a UI/meta word, a
topic/plot noun, an emoji) as sentiment, and forces a polar label from isolated tokens.
On the analysed breaks the lexical reading matched the (wrong) final label in essentially
every case. It does **not** distinguish a word that *expresses* the author's evaluation
from one merely *mentioned/referenced*.

### C. Role refinement (general principles)
- **Distinguish *expressing* from *mentioning/referencing*:** count a token as sentiment
  evidence only when it conveys the author's own evaluation, not when it names a topic,
  a UI action, or a quoted entity.
- **Do not force polarity from isolated keywords or emojis:** treat single tokens and
  emojis as *weak, defeasible* cues; require that the evaluative reading fit more than a
  lone token.
- **Report evidence *strength*, and abstain low when explicit cues are weak/absent**
  rather than over-committing to a polar label.

### D. Responsibility boundaries
Lexical **owns**: the inventory of explicit evaluation-bearing vocabulary and its
strength. It does **not** own: whether a sentiment word applies to the author vs a
mentioned third party (→ Logic), or whether discourse/tone overrides the words (→
Contextual). **When lexical cues conflict with structure or context, Lexical should
*report* the lexical reading at low confidence, not override.**

### E. Expected effect / risks
- **Reduces:** emoji-as-sentiment, UI/platform-keyword, and topic/plot-noun over-reading
  (the largest genuine-error cluster).
- **Risk:** becoming too conservative could lose the genuine wins where a strong explicit
  cue was the signal. **Mitigation:** down-weight only *weak/ambiguous* cues; keep
  surfacing *strong, unambiguous* evaluative words at full confidence.

---

## 2. Logic Agent

### A. Current role
- **Responsibility:** **rule-based/structural reasoning** — relational patterns
  (entity-action-object), concept co-occurrence, and discourse cues (enumeration,
  cause-effect, **negation**, contrast).
- **Should focus on:** structure and relationships, not surface words.
- **Should ignore:** raw vocabulary enumeration (Lexical's job).

### B. Observed failure modes (its reasoning)
The distinctive structural reasoning is **largely absent**: Logic is the **weakest**
agent (0.679) and **mirrors Lexical (96% agreement)** — it is collapsing to the same
surface read instead of resolving structure. Concretely, it fails the two errors it is
*best positioned* to catch: **sentiment-target confusion** (it should detect that "people
who disliked" is a third party, not the author) and **description-vs-evaluation**
(reported events ≠ the author's stance).

### C. Role refinement (general principles)
- **Resolve the sentiment target *first*:** identify *who* holds an attitude and *toward
  what*, before assigning any polarity. If the author is asking about or describing
  *others'* actions/feelings, the author's own sentiment may be neutral.
- **Separate reported/described content from the author's evaluation:** narrated events,
  plot, or quoted statements are not the author's sentiment.
- **Explicitly handle negation, contrast, and rhetorical/quoted constructions** for
  scope and polarity flips.

### D. Responsibility boundaries
Logic **owns**: sentiment-target attribution (author vs mentioned entity), negation/
contrast scope, and structural relations. It does **not** own: vocabulary enumeration
(Lexical) or holistic tone/pragmatics (Contextual). When structure is genuinely
ambiguous on terse text, it should **defer (low confidence) to Contextual** rather than
default to the lexical reading.

### E. Expected effect / risks
- **Reduces:** the **sentiment-target** errors (author-vs-others) and **description-as-
  sentiment** errors — i.e. the bulk of the *genuine* harm — and, by making Logic reason
  *differently* from Lexical, **decorrelates the panel** (the main systemic problem).
- **Risk:** target resolution is hard on very short/ambiguous text and could add noise.
  **Mitigation:** low confidence when the target is unclear; do not invent structure.

---

## 3. Contextual Agent

### A. Current role
- **Responsibility:** currently a **generic** "strict classification engine" (label +
  descriptions, optional prior-agent context) — the least role-specialised of the three.
  In intent, the holistic/whole-message reader.
- **Should focus on:** the message as a whole.
- **Should ignore:** —(under-specified today).

### B. Observed failure modes (its reasoning)
It is the **most accurate** agent (0.726) and the **only one that occasionally resisted**
the surface bloc — but its holistic potential is **under-used**: it still agreed with the
surface reading ~94% of the time and **missed implicit sarcasm/veiled insult/praise**
(the cases with no explicit lexical cue). Its generic prompt does not direct it to do the
pragmatic work it is best placed for.

### C. Role refinement (general principles)
- **Judge overall communicative intent and tone across the whole message**, weighing the
  message as a whole above any single token or emoji.
- **Detect sarcasm/irony** (surface polarity can invert), **implicit evaluation**, and
  **rhetorical questions** (a question can carry, or withhold, sentiment).
- **Ask explicitly whether the message *expresses* an evaluation or merely *describes/
  reacts*** — the holistic version of the mention-vs-express principle.

### D. Responsibility boundaries
Contextual **owns**: pragmatics, tone, sarcasm/irony, implicit sentiment, and the
**holistic synthesis** — and is the agent best placed to **override surface cues when the
whole-message reading contradicts them.** It should **not** re-do lexical enumeration; it
treats Lexical's evidence and Logic's target/structure as *inputs* and reasons at the
discourse level.

### E. Expected effect / risks
- **Reduces:** missed **implicit sarcasm/insult/praise**, and surface-cue traps
  (emoji-on-neutral, keyword-on-neutral) by overriding them when the whole message is
  neutral/opposite.
- **Risk:** pragmatic inference can **over-read intent** into terse text (hallucinated
  sarcasm). **Mitigation:** calibrate confidence to how much context is actually present;
  prefer neutral when the message is too short to support an inferred stance.

---

## Revised responsibility table
| agent | unique expertise | overlap (shared) | remaining gap |
|---|---|---|---|
| **Lexical** | explicit evaluation-bearing vocabulary + **cue strength / mention-vs-express** | identifying salient words (with Logic) | cannot judge *target* or *tone* |
| **Logic** | **sentiment-target attribution**, negation/contrast scope, description-vs-evaluation | discourse/structural cues (with Contextual) | does not read overall tone/pragmatics |
| **Contextual** | **pragmatics, sarcasm/irony, implicit intent, holistic synthesis & override** | some structural reasoning (with Logic) | not exhaustive on explicit vocabulary |

**Cross-cutting (apply to all three, as shared principles):** treat **emojis and UI/
platform meta-terms as weak, non-decisive cues**; **distinguish mentioning from
expressing**; **calibrate confidence** to the strength of the evidence actually present.

### Remaining framework gaps (not owned cleanly by any one agent today)
1. **Emoji / UI-artifact handling** — currently cross-cutting; the principle above
   distributes it, but no agent "owns" non-linguistic signals.
2. **Confidence calibration** — all three over-commit; calibrated low-confidence on weak
   evidence is what lets a strong primary win (and what a router needs).
3. **Mention-vs-express** — split across Lexical (token level) and Logic/Contextual
   (clause/discourse level); worth stating as a first-class shared principle.

## Diversity objective (the point of the redesign)
Today the three agents **reason the same way** (92% agreement) → effectively one
correlated vote. The redesign assigns **genuinely distinct reasoning modes** — Lexical =
*what evaluative words are present and how strong*; Logic = *who feels what toward what,
and is this description or evaluation*; Contextual = *what does the whole message
pragmatically intend (including sarcasm)*. On a case like a meta-comment about others'
"dislikes", the three would now **productively disagree** (Lexical: weak negative token;
Logic: target = others, author neutral; Contextual: rhetorical defence, neutral) instead
of unanimously over-reading. **Increasing inter-agent diversity — not making all three
reason alike — is the objective**, because diversity is what makes a consensus add value
over a single model.

(Design only — prompts unchanged. Implementation deferred until approved.)
