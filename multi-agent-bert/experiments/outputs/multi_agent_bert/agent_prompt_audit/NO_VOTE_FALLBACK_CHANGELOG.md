# No-Vote / Abstain Fallback — Implementation Changelog (audit C2)

**Implemented.** Date: 2026-06-11. Task-generic; no sentiment/topic labels
hardcoded; router unchanged; no full primary-aware consensus; tie-break
unchanged; no primary-signal prompt block; no OpenAI calls.

## Behavior change
Every former `labels[0]` fallback now **abstains** (`label=None`), which
`ConsensusAgent._extract_vote` already excludes from voting. If **all** agents
abstain, consensus **defers to the primary** (`state.primary_model_output`); if
no usable primary exists, it returns a **no-decision** (`label=None`) — **never**
`labels[0]`.

## Changed files

**New**
- `src/agents/_abstain.py` — `abstain_output(agent_name, state, reason)` helper +
  `ABSTAIN_FLAG` ("abstained"). Returns `ModelOutput(label=None, confidence=None,
  probabilities={})` with `features={"abstained": True, "abstain_reason": ...}`.

**Agents (6 fallback sites)**
- `llm_lexical_agent.py` — `_fallback_output` → `abstain_output` (+ import).
- `llm_logic_agent.py` — `_fallback_output` → `abstain_output` (+ import).
- `contextual_agent.py` — `_fallback_output` → `abstain_output` (+ import).
- `lexical_agent.py` — no-keyword-hit branch → early-return abstain (+ import).
- `logic_agent.py` — no-rule-hit branch → early-return abstain (+ import).
- `consensus_agent.py` — all-abstain branch: primary fallback, else no-decision
  (`label=None`); rationale flags `primary_fallback` / `no_decision`. (Used the
  already-imported `Optional`; reads `state.primary_model_output`.)

**Tests**
- New: `tests/test_abstain_fallback.py` (11 tests) — helper shape; LLM
  parse-failure & invalid-label abstain (lexical+logic); contextual parse-failure
  abstain; non-LLM no-hit abstain (lexical+logic); consensus ignores abstentions;
  all-abstain → primary; all-abstain-no-primary → None (never labels[0]).
- Updated to the abstain contract: `test_lexical_agent.py`, `test_logic_agent.py`,
  `test_llm_specialist_agents.py`, `test_contextual_agent.py`,
  `test_consensus_agent.py` (former first-label-fallback assertions → abstain /
  primary-fallback / no-decision).

## Test results
- **Full offline suite: 871 passed** (was 859; +new abstain tests, renamed
  updates). No OpenAI, no downloads.

## Metric change (offline; paper_style is the only mode that changes measurably)
paper_style, XLM-R primary, threshold 0.9, EESA test (free, local):

| Metric | BEFORE | AFTER | Δ |
|---|---|---|---|
| Accuracy | 0.7408 | 0.7421 | +0.0013 |
| Macro F1 | 0.7163 | 0.7173 | +0.0010 |
| Escalated acc | 0.2000 | 0.2053 | +0.0053 |
| Escalation rate | 0.2323 | 0.2323 | 0 (router unchanged) |
| pos / neg / neu F1 | 0.795 / 0.662 / 0.691 | 0.797 / **0.662** / 0.693 | +/0/+ |

**No regression**; tiny net improvement (removing the first-label/positive bias on
no-match). Negative F1 unchanged.

`full_agentic` + real GPT-4o-mini was **not** re-run (constraint: no OpenAI). With
clean JSON output the LLM fallbacks rarely fire, so the expected effect there is
near-zero on headline metrics; the value is robustness (no positive collapse on
API hiccups) — to be confirmed in a later paid validation if desired.

## Constraints honored
Task-generic (no label names); router unchanged; only the minimal all-abstain→
primary added (not full primary-aware consensus C1); tie-break unchanged; no
primary-signal block; no OpenAI calls.
