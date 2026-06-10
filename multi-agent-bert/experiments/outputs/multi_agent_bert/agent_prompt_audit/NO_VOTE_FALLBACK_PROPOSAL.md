# Design Proposal — No-Vote / Abstain Fallback (audit C2)

**Proposal only — nothing implemented, no OpenAI calls.** Replaces every
`labels[0]` fallback in the classification flow with a task-generic **abstain**
(no-vote) mechanism; the only primary-awareness added is the minimal all-abstain
→ primary fallback (req 7/10). Router unchanged. No sentiment/topic labels
hardcoded. Date: 2026-06-10.

## Key enabler (no schema change needed)
- `ModelOutput.label`, `ConsensusOutput.label`, `FinalOutput.label` are already
  `Optional[str] = None` ([schema.py:79](../../../src/state/schema.py#L79)).
- `ModelOutput.validate_labels` **skips** label validation when `label is None`
  ([schema.py:88](../../../src/state/schema.py#L88)) — so an abstain output is
  schema-valid (keep `probabilities={}` so its prob-keys also pass).
- `ConsensusAgent._extract_vote` already returns `(None, None)` for a None label
  → the agent is **already excluded** from voting
  ([consensus_agent.py:68-75](../../../src/agents/consensus_agent.py#L68)).

So "abstain" = an `AgentOutput` whose `model_output.label is None`. Most of the
plumbing exists; the change is making fallbacks *produce* that instead of
`labels[0]`, plus fixing the all-abstain branch.

---

## 1–3. Current `labels[0]` fallback sites — what they do / what they should return

| # | File:line / function | Current behavior | Proposed |
|---|---|---|---|
| 1 | `llm_lexical_agent.py:209` `_fallback_output` | parse/invalid-label → emits `labels[0]`, uniform conf, uniform probs (**votes for first label**) | **abstain** (label None, conf None, probs {}) |
| 2 | `llm_logic_agent.py:208` `_fallback_output` | same | **abstain** |
| 3 | `contextual_agent.py:306` `_fallback_output` | same (parse failure) | **abstain** |
| 4 | `lexical_agent.py:109` (no keyword hit) | `best_label = labels[0]`, uniform probs (**votes first label** when no keyword matches) | **abstain** (no lexical signal → no vote) |
| 5 | `logic_agent.py:118` (no rule hit) | `best_label = labels[0]`, uniform probs | **abstain** (no rule signal → no vote) |
| 6 | `consensus_agent.py:207` (no usable votes) | final = `labels[0]`, uniform conf (**all-abstain collapses to first label**) | **fall back to primary** prediction (req 7) |

Sites 1–3 are the `full_agentic` LLM agents (highest priority — they were the
positive-collapse path under the mock). Sites 4–5 are the `paper_style` non-LLM
agents (include for consistency; this *does* change paper_style behavior — see
risks). Site 6 is the critical all-abstain case.

## 4. Abstain / no-vote representation

A shared helper (proposed `abstain_output(agent_name, state, reason)` in a small
`src/agents/_abstain.py`, or a `BaseAgent` classmethod — keeps it DRY across 5
agents):

```python
AgentOutput(
    agent_name=agent_name,
    model_output=ModelOutput(label=None, confidence=None, probabilities={},
                             raw_text=state.input_text),
    notes=reason,                       # e.g. "parse failure", "no lexical match"
    features={"abstained": True, "abstain_reason": reason},
)
```

- `label=None` + `confidence=None` → excluded by `_extract_vote` (defense in
  depth: either being None suffices).
- `probabilities={}` → no fake uniform distribution; passes `validate_labels`.
- `features["abstained"]` → an explicit, queryable flag (for logging,
  explainability, and tests) without relying on parsing `notes`.
- **Fully task-generic** — no label names anywhere.

## 5. How ConsensusAgent ignores abstentions
Already handled: `_extract_vote` returns `(None, None)` → the loop hits
`if label is None ... continue` and records `vote_details[slot] = "no vote"`
([consensus_agent.py:160](../../../src/agents/consensus_agent.py#L160)). No
change to the per-agent path. `active_weight_sum` only accumulates real votes, so
abstainers don't dilute the denominator either.

## 6–7. All-abstain behavior → fall back to primary
Replace the `active_weight_sum == 0` branch
([consensus_agent.py:205-229](../../../src/agents/consensus_agent.py#L205)):

```
if active_weight_sum == 0.0:
    primary = state.primary_model_output            # already in state on escalation
    if primary is not None and primary.label is not None \
       and task.is_allowed_label(primary.label):
        label, conf, source = primary.label, primary.confidence, "primary_fallback"
    else:
        label, conf, source = None, None, "abstain"   # NEVER labels[0]
    # write consensus_output + final_output with (label, conf), rationale notes source
```

- This is the **only** primary-awareness in this fix (req 10) — a degenerate
  safety net, not the full primary-aware consensus (C1, separate step).
- In the classification escalation path the router runs **after** the primary, so
  `primary_model_output` is essentially always present → all-abstain keeps the
  (strong) primary instead of defaulting to `positive`.
- The `label=None` "true abstain" final only occurs if the primary is somehow
  missing/invalid (misconfiguration) — and even then we **never** invent
  `labels[0]`.

Tie-break is **unchanged** in this fix (the positional tie-break M2 is a separate
proposal). With abstains there are simply fewer votes; ties among real votes
still resolve as today.

---

## Affected files / classes / functions
- **Change:** `llm_lexical_agent.py` (`_fallback_output`), `llm_logic_agent.py`
  (`_fallback_output`), `contextual_agent.py` (`_fallback_output`),
  `lexical_agent.py` (no-hit branch), `logic_agent.py` (no-hit branch),
  `consensus_agent.py` (all-abstain branch).
- **New:** `src/agents/_abstain.py` (shared `abstain_output` helper) or a
  `BaseAgent` classmethod.
- **Touch (graceful rendering of abstainers):** `explainability_agent.py` /
  `llm_explainability_agent.py` and `contextual_agent._build_prior_summaries` —
  render an abstained upstream agent as `"<agent>: abstained (reason)"` instead
  of skipping silently or printing a fake label. Low effort; explanation-only.
- **No change:** router, schema, config, NER agents/path, prompts.

## Proposed data representation
See §4 — `label=None`, `confidence=None`, `probabilities={}`,
`features={"abstained": True, "abstain_reason": str}`. Optionally add a module
constant `ABSTAIN_NOTE`/flag key for consistent assertions.

## Edge cases
- **Some vote, some abstain** → abstainers excluded; winner from the real votes
  (already supported).
- **Exactly one agent votes** → consensus returns that label (fine).
- **All abstain, primary present** → final = primary label/confidence (the common
  safety case).
- **All abstain, primary missing/invalid** → final `label=None`, rationale =
  "all agents abstained; no primary available"; **evaluator must treat a None
  final label as an error sample**, not as a prediction (today it would read
  None — needs a guard so it doesn't crash or silently miscount).
- **Abstain output with empty `probabilities`** → any downstream consumer
  expecting a distribution must tolerate `{}` (consensus already does; check
  explainability/serialization).
- **Deliberation enabled** (off by default) → an abstaining specialist simply
  isn't in the vote tuples; deliberation prompt should show "abstained".

## Expected impact — sentiment
- Removes the **positive** bias on parse failures / no-match / all-abstain (the
  exact mechanism behind the mock collapse and the connection-error contamination).
- On **clean `full_agentic` + real GPT-4o-mini**: near-zero effect on headline
  numbers (0 parse errors today → fallbacks rarely fire), but makes results
  **robust to API hiccups** and removes a latent bias.
- On **`paper_style`**: a real behavior change — lexical/logic now abstain when
  they find no keyword/regex hit (instead of voting `positive`); if all abstain,
  defer to the (real transformer) primary. Likely neutral-to-positive for
  accuracy, but must be measured (it's free — paper_style uses no API).

## Expected impact — future topic classification
- Larger benefit: removes first-label (`business`) bias, which is more harmful
  across 9 classes; and all-abstain→primary protects the strong primary when the
  9-way agents are unsure. Fully generic (label-name-free), so it transfers
  unchanged.

## Risks
- **Low overall.** Main risks:
  1. **paper_style shift** (sites 4–5) — could move paper_style metrics; validate
     with a local paper_style sentiment run (no API, free).
  2. **Test updates** — several tests assert the old behavior and must change:
     `test_contextual_agent.py` (`test_echo_fallback_to_first_label_when_none_found`,
     `test_fallback_label_is_first_in_list`), `test_logic_agent.py`
     (`test_fallback_picks_first_label`), `test_llm_specialist_agents.py`
     (invalid-label fallback tests), and `test_consensus_agent.py` (no-vote
     rationale + all-abstain now → primary, not `labels[0]`). (Fixture-only uses
     of `labels[0]` in test_evaluator/test_transformer_contextual are *not*
     fallback assertions — leave them.)
  3. **Evaluator None-final guard** — add handling so a true-abstain final
     (primary missing) counts as an error sample, never `labels[0]`.
  4. **Explainability rendering** of abstained agents (cosmetic).

## Implementation sequence (when approved)
1. Add the shared `abstain_output` helper + `features["abstained"]` flag (no
   behavior change yet).
2. Switch the 5 agent fallbacks (sites 1–5) to abstain.
3. Replace consensus all-abstain branch with primary fallback (site 6) + None-final
   guard semantics.
4. Make explainability / prior-summaries render abstained agents gracefully.
5. Update/extend tests (below).
6. Verify: full offline suite green; run a **local paper_style** sentiment eval
   (free, no API) to confirm no regression; defer the `full_agentic` real-LLM
   confirmation to the validation step (small paid EESA run) — **not** part of
   this PR.

## Tests needed (all offline, no OpenAI)
- **Agents abstain correctly:** for each of `llm_lexical`, `llm_logic`,
  `contextual` — a forced parse failure (MockLLMClient `raise_on_call` / garbage
  fixed response) yields `model_output.label is None`, `confidence is None`,
  `probabilities == {}`, `features["abstained"] is True`.
- **Non-LLM no-match:** `lexical_agent` with empty keyword_map and `logic_agent`
  with empty rule_map → abstain (label None), not `labels[0]`.
- **Consensus ignores abstainers:** mix of one real vote + two abstains → winner =
  the real vote; abstainers shown as "no vote".
- **All-abstain → primary:** all three abstain, `state.primary_model_output` set →
  `final_output.label == primary.label`, rationale flags `primary_fallback`.
- **All-abstain, no primary:** → `final_output.label is None` (never `labels[0]`),
  rationale notes abstain; evaluator counts it as an error sample.
- **No positional leakage:** with all abstains + a primary whose label is *not*
  `labels[0]`, the final is the primary's label (guards against re-introducing
  first-label bias).
- **Update** the existing first-label-fallback assertions listed under Risks #2.
