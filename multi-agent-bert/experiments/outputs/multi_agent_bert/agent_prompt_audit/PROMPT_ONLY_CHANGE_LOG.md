# Prompt-only change — generic classification wording

**Implemented.** Date: 2026-06-10. Safe, prompt-only cleanup so the shared
classification flow reads as task-generic (works for **sentiment and topic**
without rewording). No behaviour, schema, agent, consensus, router, fallback, or
state change.

## What changed (2 files)

### `src/prompts/llm_lexical_prompt.py`
- System prompt: "identify the most likely **topic** label" → "choose the most
  likely **classification label for the ACTIVE TASK**"; "**Domain-specific**
  terminology" → "**Task-relevant** terminology … (in any language present, e.g.
  Arabic and English)"; "Named entities and technical terms" → "Named entities
  and salient tokens"; rule 2 now anchors on "the **LABEL DESCRIPTIONS** for the
  active task".
- User template: "identify **domain-specific** vocabulary in both Arabic and
  English" → "identify **task-relevant** vocabulary (in any language present …)".

### `src/prompts/llm_logic_prompt.py`
- System prompt: "determine the most likely **topic** label" → "choose the most
  likely **classification label for the ACTIVE TASK**"; "**domain-relevant**
  concept pairs" → "**task-relevant** concept pairs (in any language present)";
  discourse cues now "enumeration, cause-effect, **negation, and contrast**"
  (generic discourse phenomena, not sentiment wording); "Reason about what
  **domain** this text belongs to" → "Reason about **which allowed label** best
  fits the text for the active task"; rule 2 anchors on the label descriptions.
- User template: "… in both Arabic and English that point to **one domain**" →
  "(in any language present …) that point to **one allowed label**".

## What did NOT change
- JSON schemas (still `label`, `confidence`, `reasoning`, `evidence`) and strict
  JSON-only output.
- `build_user_prompt` signatures (no new params; no primary-signal block yet).
- `contextual_prompt.py`, `llm_explainability_prompt.py` — untouched.
- ConsensusAgent, Router, fallbacks (`labels[0]`), agent state, config — untouched.
- No sentiment labels and no topic labels hardcoded; labels + label_descriptions
  still injected from `task_config`.

## Verification
- **Full offline suite: 859 passed** (unchanged — no test depended on the old
  wording).
- Grep: **0** occurrences of "topic"/"domain" remain in the two prompt files.
- Offline render for a sentiment config: both prompts now say "classification
  label for the ACTIVE TASK", inject `positive, negative, neutral` from config,
  and keep the JSON-only instruction.
- **Real-LLM smoke: not run** — the change is wording-only with an unchanged JSON
  contract already proven in the pilot/sweep (JSON mode held, 0 parse errors), so
  a paid smoke adds no signal (and the connection has been unstable). Can be run
  on request.

## Deferred to follow-up steps (not in this PR)
Optional primary-signal block, no-vote (abstain) fallback, primary-aware
consensus, non-positional tie-break, per-task / margin-based router — each its
own change, to be sequenced when the fix order is agreed.
