# Primary-Aware Consensus — Implementation Changelog (audit C1, Fix #2)

**Implemented (simplest version).** Date: 2026-06-11. Task-generic; no
sentiment/topic labels hardcoded; router unchanged; prompts unchanged; no
primary-signal block; agent confidences unchanged; Fix #1 abstain behavior
preserved; **no OpenAI calls**.

## Behavior change
On the escalation path, when ≥1 agent votes, the **primary now participates as a
confidence-scaled weighted vote**: `scores[primary.label] += w_primary *
primary.confidence`, with default **`w_primary = 1.0`**. Agents override the
primary only when an alternative label's combined agent score exceeds the
primary's contribution. Tie-breaking is now **non-positional**. The all-abstain
case still defers to the primary (Fix #1, unchanged — handled before the primary
vote is injected, so no double-count).

## Tie-break (config-order-free), via new `_select_winner`
1. primary's label if it is tied → choose primary (conservative anchor);
2. else label with more voting agents;
3. else label with the highest single agent contribution;
4. else deterministic **alphabetical** (NOT `task_config.labels` order).

## Changed files
- `src/agents/consensus_agent.py`:
  - `_DEFAULT_WEIGHTS` gains `"primary": 1.0`; `__init__` clamps/keeps the
    `"primary"` slot. `ConsensusAgent(weights={"primary": X})` accepts
    0 / 1.0 / 1.5 / 2.0 (programmatic).
  - `run()` tracks `vote_counts` + `max_contribution` per label; injects the
    primary vote after the all-abstain guard; selects the winner via
    `_select_winner` (non-positional).
  - New module-level `_select_winner(tied, primary_label, vote_counts, max_contribution)`.
- Tests: `tests/test_consensus_agent.py` — updated `test_default_weights_are_equal`
  (now includes `primary: 1.0`) and the two positional-tie tests →
  non-positional assertions. New `tests/test_primary_aware_consensus.py` (10
  tests): primary counted; single agent can't flip a confident primary; two
  agents flip a low-confidence primary; low-vs-high-confidence override
  sensitivity; `w_primary=0` reproduces agents-only; tie prefers primary
  (not labels[0]); tie without primary uses agent-count; all-abstain→primary
  preserved; arbitrary-label task-generic.

> Note: existing consensus tests built via `make_state()` set **no primary**
> (`primary.label is None`), so the primary vote is skipped there — they stayed
> valid without edits except the two that asserted the old positional tie-break.

## Test results
**Full offline suite: 881 passed** (was 871; +10 new). No OpenAI, no downloads.

## Metric change — paper_style (the mode expected to benefit most; free/local)
XLM-R primary, threshold 0.9, EESA test:

| Metric | BEFORE (Fix #1) | AFTER (Fix #2) | Δ |
|---|---|---|---|
| Accuracy | 0.7421 | **0.8056** | **+0.0635** |
| Macro F1 | 0.7173 | **0.7882** | **+0.0709** |
| Escalated accuracy | 0.2053 | **0.4789** | **+0.2736 (2.3×)** |
| Escalation rate | 0.2323 | 0.2323 | 0 (router unchanged) |
| pos / neg / neu F1 | 0.797 / 0.662 / 0.693 | **0.883 / 0.739 / 0.743** | +0.086 / **+0.077** / +0.050 |

**paper_style improves clearly and on every class** (negative +0.077, no
regression). Reference: primary_only XLM-R = 0.8240 / 0.8088 — paper_style now
0.8056 / 0.7882, i.e. the weak keyword/regex/TF-IDF agents no longer drag the
strong primary down; they are anchored to it and only override when they
out-vote it. This is the predicted outcome (weak deterministic agents benefit
most from primary anchoring).

## Not run (constraint)
`full_agentic` + real GPT-4o-mini was **not** re-run (no-OpenAI constraint). The
real-LLM agents are *good* on the escalated slice, so primary anchoring there is
expected to slightly temper the upside while cutting regressions — net depends on
`w_primary`. A **single small paid EESA run** (sweep `w_primary` ∈ {1.0, 1.5, 2.0}
+ the `w_primary=0` baseline) is the recommended follow-up to tune it; awaiting
explicit approval.

## Follow-up (small)
To sweep `w_primary` via `evaluate_pipeline.py`, wire `build_orchestrator` to
read `execution.consensus_weights.primary` from config and pass it to
`ConsensusAgent(weights=...)` (the deliberation weight is likewise currently
unwired). Default stays 1.0; not required for the paper_style validation above.

## Constraints honored
Task-generic (no label names) · router unchanged · prompts unchanged · no
primary-signal block · agent confidences/calibration unchanged · Fix #1 abstain
preserved · default `w_primary = 1.0` · no OpenAI calls.
