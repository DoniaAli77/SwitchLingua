# Prompt-Improvement Proposal (classification flow)

**Proposal only — nothing implemented.** Targets the shared classification flow
(sentiment **and** topic). NER untouched. No sentiment/topic labels hardcoded;
all task-specific values keep coming from `task_config`. Date: 2026-06-10.

## Scope decision per file

| Prompt file | Change? | Reason |
|---|---|---|
| `llm_lexical_prompt.py` | **Yes — reword** | system prompt says "topic label", "domain-specific" |
| `llm_logic_prompt.py` | **Yes — reword** | system prompt says "topic label", "what domain this text belongs to" |
| `contextual_prompt.py` | **Wording: no.** Optional block: yes | already generic ("strict text classification engine"); only needs the shared optional primary block |
| `llm_explainability_prompt.py` | **No change** | already task-agnostic and **explanation-only** (does not affect the label/accuracy); not worth touching now |

All four already inject `labels` + `label_descriptions` from `task_config` — that
part stays. The JSON contracts stay **unchanged** (strict JSON, same keys).

---

## A. `llm_lexical_prompt.py` — SYSTEM_PROMPT

**Before (topic-flavored):**
> Your role is to identify the most likely **topic** label based on VOCABULARY CUES ONLY:
> - … - **Domain-specific** terminology (in both Arabic and English)

**Proposed (generic):**
```
You are a lexical analysis specialist in a multi-agent text classification system.

Your role is to choose the most likely classification label for the ACTIVE TASK,
based on VOCABULARY CUES ONLY:
- Surface-level words, terms, and phrases that appear explicitly in the text
- Task-relevant terminology and characteristic expressions (in any language
  present in the text, e.g. Arabic and English)
- Named entities and salient tokens

RULES — follow every rule exactly:
1. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
2. Base the decision on explicit lexical evidence — words/phrases visible in the
   text — matched against the LABEL DESCRIPTIONS for the active task.
3. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
4. The JSON must contain exactly these four keys:
   - "label"      : string — one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, 0.0–1.0
   - "reasoning"  : string — one sentence citing the key vocabulary you found
   - "evidence"   : array  — 1–5 tokens or short phrases from the text

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```
Changes: "topic label" → "classification label for the ACTIVE TASK";
"domain-specific terminology" → "task-relevant terminology"; added "matched
against the LABEL DESCRIPTIONS for the active task" so semantics ride on config,
not on the word "topic". User-template wording ("identify domain-specific
vocabulary") → "identify task-relevant vocabulary".

## B. `llm_logic_prompt.py` — SYSTEM_PROMPT

**Before:**
> determine the most likely **topic** label … - Reason about what **domain** this text most logically belongs to

**Proposed (generic):**
```
You are a logical reasoning specialist in a multi-agent text classification system.

Your role is to choose the most likely classification label for the ACTIVE TASK
by applying RULE-BASED AND STRUCTURAL REASONING:
- Identify relational patterns between concepts (e.g. entity-action-object)
- Detect co-occurrence of task-relevant concept pairs (in any language present)
- Apply discourse-level cues: enumeration, cause-effect, negation, and contrast
- Reason about which allowed label best fits the text for the active task

RULES — follow every rule exactly:
1. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
2. Base the decision on logical inference — patterns and relationships, not just
   surface words — matched against the LABEL DESCRIPTIONS for the active task.
3. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
4. [same four keys: label, confidence, reasoning, evidence]

OUTPUT FORMAT:
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```
Changes: "topic label"/"what domain" → "classification label for the ACTIVE
TASK"/"which allowed label best fits". Added **negation** and **contrast** to the
discourse cues — these are generic discourse phenomena (not sentiment-specific
wording) that happen to help polarity tasks without hardcoding them.

## C. `contextual_prompt.py`
**No wording change** (already generic). Only extend its user template with the
shared optional primary block (§E) for parity with lexical/logic.

## D. `llm_explainability_prompt.py`
**No change proposed.** It is already task-agnostic, and it writes only
`explanation_output` — it never changes the final label or accuracy. Leave it
out of this round.

---

## E. Optional "primary model signal" block (shared by lexical / logic / contextual)

A new **optional, config-gated** block injected near the bottom of each
classifier's user prompt. Default = empty string → **fully backward compatible**.
When enabled it turns the agents from blind re-classifiers into **adjudicators**
of an uncertain primary (addresses audit M5 / "ChatGPT Issue 2").

**Proposed rendered block:**
```
PRIMARY MODEL SIGNAL (context only — adjudicate, do NOT simply copy it):
  predicted label : $p_label
  confidence      : $p_conf
  top-2 labels    : $p_top2            e.g. "negative (0.43), neutral (0.39)"
  full distribution: $p_dist           e.g. "positive=0.18, negative=0.43, neutral=0.39"
Weigh your own analysis. Agree with the primary only if the text evidence
supports it; if the evidence points elsewhere, choose the better-supported label.
```
Design notes:
- All four requested fields included: **primary label, confidence, full
  probabilities, top-2 labels.** Top-2 is derived from the probability vector.
- The "do NOT simply copy" framing is deliberate — it mitigates **anchoring**
  (the risk that agents just echo the primary, collapsing the ensemble). Keep it
  optional so we can A/B independent-vs-adjudicator behaviour.
- Most informative with the **real transformer primary** (it emits a genuine
  probability distribution); the mock primary's probabilities are less
  meaningful, so default the flag off for mock runs.
- JSON output contract is unchanged.

---

## F. Prompt-only vs. agent-state changes

| Change | Prompt-only? | Needs agent edit? | Needs new state field? |
|---|---|---|---|
| Reword lexical/logic system prompts (topic/domain → generic) | ✅ **prompt-only** | No | No |
| Add `$primary_block` placeholder to templates, default-empty render | ✅ **prompt-only** (backward compatible) | No | No |
| **Populate** the primary block at runtime | ❌ | ✅ agents must read state + compute top-2 + pass `primary_signal` to `build_user_prompt` | **No** — data already exists |
| Config flag to gate the block (e.g. `execution.agents_use_primary_signal`) | ❌ | ✅ loader + agent read | No |

Key point: **the primary's label, confidence, and probabilities already live in
`state.primary_model_output`** (the transformer writes them today). So the
optional block needs **no new state field** — only:
1. the prompt templates to accept an optional `primary_signal` argument (and
   render the block or empty), and
2. the three classifier agents' `run()` to extract `(label, confidence,
   probabilities)` from `state.primary_model_output`, compute the top-2, and pass
   them through (gated by a config flag).

So the work splits cleanly:
- **Pure prompt PR** (safe, no behavior change): the wording fixes + adding the
  optional placeholder that defaults to empty. Existing tests/behaviour unchanged.
- **Follow-up agent PR** (opt-in behaviour change): wire the agents + config flag
  to actually fill the block, then evaluate independent-vs-adjudicator.

---

## G. Requirements check
- ✅ "topic/domain" → "classification label / active task / task-relevant cues".
- ✅ labels + label_descriptions still injected from `task_config`.
- ✅ no positive/negative/neutral hardcoded; no topic labels hardcoded.
- ✅ optional primary-uncertainty block: label, confidence, probabilities, top-2.
- ✅ strict JSON output preserved (schemas unchanged).
- ✅ prompt-only vs agent-state changes separated (§F).

## H. Out of scope here (separate proposals, per the audit)
Not part of this prompt round: removing `labels[0]` fallbacks (C2),
primary-aware consensus / non-positional tie-break (C1/M2), per-task /
margin-based router (M4). Those are agent/consensus/router code changes, to be
sequenced after the prompt PR when we agree the fix order.
