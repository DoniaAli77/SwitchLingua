# Optional Primary-Signal Prompt Block — Implementation Changelog (Fix #3)

**Implemented, default OFF, fully config-gated.** Date: 2026-06-11. No OpenAI
calls. JSON schema unchanged; router unchanged; consensus unchanged; no label
hardcoding. Current behavior is identical unless `agents_use_primary_signal` is
explicitly enabled.

## What it does
When `task_config.agents_use_primary_signal` is True, `LLMLexicalAgent`,
`LLMLogicAgent`, and `ContextualAgent` prepend a **primary-signal block** to their
user prompt (primary label, confidence, top-2 labels, full distribution) with
explicit **anti-anchoring** wording. Default off → block renders as the empty
string and the prompt is byte-identical to before.

## Changed files
- **New** `src/prompts/_primary_block.py`:
  - `build_primary_signal(primary_output) -> dict | None` — `None` when no usable
    label; `top2` = two highest-probability `(label, prob)` pairs (by probability,
    never label order).
  - `render_primary_block(signal, analysis_kind) -> str` — renders the block or
    `""`; degrades gracefully when probabilities are missing; `analysis_kind`
    ("lexical"/"logical"/"contextual") is used only in the anti-anchoring sentence.
- `src/prompts/llm_lexical_prompt.py`, `llm_logic_prompt.py`, `contextual_prompt.py`:
  optional `$primary_block` placeholder + `primary_signal` param on
  `build_user_prompt` (default None → empty). No other wording changed.
- `src/agents/llm_lexical_agent.py`, `llm_logic_agent.py`, `contextual_agent.py`:
  build `primary_signal` from `state.primary_model_output` and pass it **only when
  the flag is on**; else `None`.
- `src/state/schema.py`: `TaskConfig.agents_use_primary_signal: bool = False`.
- `src/config/loader.py`: reads `execution.agents_use_primary_signal`.
- `src/config/default.yaml`: `execution.agents_use_primary_signal: false`.
- **Tests** `tests/test_primary_signal_block.py` (12): top-2 sorted by probability
  (not label order); block contains signal + anti-anchoring; None/empty-prob/no-label
  safe; prompts omit block without a signal and include it with one; **agents omit
  block when flag off / include when on**; flag-on-but-no-primary → no block, no
  crash; **JSON schema unchanged with block on**; task-generic arbitrary labels.

## Anti-anchoring wording (rendered)
> PRIMARY MODEL SIGNAL (context only — you are an independent {kind} adjudicator):
> … The primary may be wrong — especially when its confidence is low or its top-2
> are close. Do your own {kind} analysis of the text FIRST. Use this signal as
> context only; … Do NOT simply copy the primary.

## Missing-probabilities handling
- `primary.label is None` → empty block (agent runs blind).
- label present, `probabilities` empty → render label + confidence, omit top-2 /
  distribution, add "(probability distribution unavailable)".
- `confidence is None` → printed as "n/a".

## Test results
**Full offline suite: 893 passed** (was 881; +12 new). No OpenAI, no downloads.
Default off → every existing test unchanged.

## Not run
No paid experiments (per instruction). The block's value is uncertain (anchoring
risk) and is the subject of the 2×2 ablation — see
`ABLATION_PLAN_2x2.md` (awaiting approval before any paid run).

## Constraints honored
Task-generic (no label names) · default OFF · JSON schema unchanged · router
unchanged · consensus unchanged · missing-probabilities safe · no OpenAI.
