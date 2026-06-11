# Design Proposal — Optional Primary-Signal Prompt Block (audit M5, Fix #3)

**Proposal only — nothing implemented, no OpenAI calls.** Lets the three LLM
classifier agents optionally see the primary model's prediction (label,
confidence, top-2, full distribution) so they **adjudicate** an uncertain primary
instead of classifying blind. Config-gated, **default OFF**, renders empty when
disabled. Consensus/router/JSON schema unchanged; no label hardcoding. Date:
2026-06-11.

## Affected agents (only the LLM classifiers, full_agentic)
`LLMLexicalAgent`, `LLMLogicAgent`, `ContextualAgent`. The non-LLM paper_style
agents (`LexicalAgent`, `LogicAgent`, `TransformerContextualAgent`) and the NER
agents are **untouched** → **paper_style is unaffected**.

---

## 9. Prompt files that change (3)
- `src/prompts/llm_lexical_prompt.py`
- `src/prompts/llm_logic_prompt.py`
- `src/prompts/contextual_prompt.py`

Each gets:
1. an optional `$primary_block` placeholder in `_USER_TEMPLATE`, just above the
   final "Respond with JSON only…" line;
2. a new optional param on `build_user_prompt(..., primary_signal: dict | None = None)`
   that renders the block when a signal is supplied, else the empty string.

**New shared helper** `src/prompts/_primary_block.py` (avoids 3× duplication):
```python
def render_primary_block(signal: dict | None, analysis_kind: str) -> str:
    """Render the primary-signal block, or '' when signal is None/unusable."""
```
`analysis_kind` is the agent's own role word ("lexical" / "logical" /
"contextual") so the anti-anchoring sentence reads naturally — it is **not** a
task label, so this stays task-generic.

### Proposed rendered block (anti-anchoring; req 13)
```
PRIMARY MODEL SIGNAL (context only — you are an independent {analysis_kind} adjudicator):
  predicted label  : {label}
  confidence       : {conf:.2f}
  top-2 labels     : {l1} ({p1:.2f}), {l2} ({p2:.2f})
  full distribution: {lbl=prob, ...}
The primary may be wrong — especially when its confidence is low or its top-2 are
close. Do your own {analysis_kind} analysis of the text FIRST. Agree with the
primary only if the text evidence supports its label; choose a different allowed
label if the evidence points elsewhere. Do NOT copy the primary by default.
```
- No JSON-schema change (req 6): this is **input context only**; the required
  output keys (`label, confidence, reasoning, evidence`) are unchanged.

## 10. Agent files that change (3) + config
- `llm_lexical_agent.py`, `llm_logic_agent.py`, `contextual_agent.py`: in `run()`,
  when the flag is on, build a `primary_signal` dict from
  `state.primary_model_output` and pass it to `build_user_prompt(...,
  primary_signal=...)`. When off → pass `None` → empty block → **no behavior
  change**.
- `src/state/schema.py`: add `agents_use_primary_signal: bool = False` to
  `TaskConfig` (mirrors the existing `contextual_use_prior_outputs` flag).
- `src/config/loader.py`: read `execution.agents_use_primary_signal` (+ optional
  override param, like the other flags).
- `src/config/default.yaml`: `execution.agents_use_primary_signal: false`.
- (Optional) `evaluate_pipeline.py`: a `--agents_use_primary_signal` CLI flag for
  A/B convenience. Not required.

**No change** to consensus, router, NER, explainability, or the output schema.

## 11. How top-2 is computed
From `state.primary_model_output.probabilities` (a `{label: prob}` dict the
transformer primary already populates over the task labels):
```python
probs = dict(primary.probabilities)
ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)  # desc by prob
top2 = ranked[:2]            # up to two (label, prob) pairs
```
Task-generic — sorts by probability value, never by label name/order. The
`primary_signal` dict passed to the prompt:
```python
{"label": primary.label, "confidence": primary.confidence,
 "top2": top2, "distribution": probs}
```

## 12. Handling missing probabilities
`render_primary_block` degrades gracefully:
- **No usable primary label** (`primary.label is None`) → return `""` (agent runs
  blind, as today). Unreachable on the escalation path (primary always ran), but
  safe.
- **Label present, probabilities empty/None** (e.g. a mock primary) → render
  `predicted label` + `confidence` only; **omit** the top-2 and distribution lines
  and add "(probability distribution unavailable)". `top2 = []`.
- **Exactly one label / one prob** → render the single top entry; no crash.
- Numbers formatted defensively (guard `None` confidence → "n/a").

## 13. Anti-anchoring (see block text above)
The block explicitly frames the signal as **context, not an answer**: "you are an
independent adjudicator", "do your own analysis FIRST", "the primary may be
wrong", "do NOT copy by default", and it highlights *when* to distrust it (low
confidence / close top-2). Kept **off by default** so it's opt-in and A/B-able.

## 14. Tests needed (all offline, no OpenAI)
- **`render_primary_block`:** full signal → contains label/conf/top-2/distribution
  + the anti-anchoring sentence; missing-probabilities → label/conf only, no crash;
  `primary.label is None` → returns `""`; single-label edge.
- **top-2 util:** correct descending order; ties stable; <2 labels handled; never
  uses label order.
- **Flag gating (the key behavioral test):** run an LLM agent with a recording
  `MockLLMClient` (capture `call_log`); with `agents_use_primary_signal=False` the
  prompt **does not** contain "PRIMARY MODEL SIGNAL"; with it `True` (and a
  primary set in state) the prompt **does**, and includes the primary label.
- **No-op when off:** `TaskConfig` default flag is `False`; agent output identical
  to pre-Fix-#3 (snapshot the rendered prompt).
- **No JSON-schema change:** with the flag on, the agent still parses a normal
  `{label, confidence, reasoning, evidence}` response (label_echo mock).
- **Task-generic:** arbitrary labels `["a","b","c","d"]` render correctly.
- **Missing-primary path:** flag on but `primary.label is None` → empty block, agent
  still produces a valid (or abstaining) output.

## 15. Expected impact
- **Sentiment — full_agentic real-LLM (the only affected mode):** agents see the
  primary's actual confusion (e.g. "negative 0.43 / neutral 0.39") and can target
  it → *potential* lift on the escalated slice beyond Fix #2. Counter-force is
  **anchoring** (agents echo the primary), which would reduce the independent
  signal. Net is genuinely unknown → **paid A/B (flag off vs on) required**;
  expected small at threshold 0.6, larger at high thresholds.
- **Topic classification (future):** likely the bigger beneficiary — among 9
  classes the top-2 focuses agents on the two real contenders rather than guessing
  across all nine; same anchoring risk; fully generic, transfers unchanged.
- **paper_style:** **no effect** — non-LLM agents don't receive the block.
- **full_agentic real-LLM:** the sole mode where behavior changes (when enabled).

## 16. Risks
1. **Anchoring to the primary (primary risk).** Agents may copy the primary
   instead of independently judging, collapsing ensemble diversity and **negating
   the value of the agents** (they'd stop catching primary errors). Mitigations:
   strong anti-anchoring wording; **default off**; A/B test; measure
   *agent-vs-primary agreement rate* before/after — a large jump signals anchoring.
2. **Interaction with Fix #2 (compounding bias).** Consensus *already* anchors to
   the primary (w_primary). Showing the primary to the agents too is a **second**
   anchor → the pipeline could over-trust the primary and drift toward
   primary_only. Must A/B with Fix #2 active; if it over-anchors, the remedy is to
   **lower w_primary when the block is on** (tune them together), not to hardcode.
3. **Confident-but-wrong primary.** Displaying high confidence may push agents to
   agree with a wrong primary. The anti-anchoring "may be wrong even when
   confident… check the evidence" wording partially guards this.
4. **Mock primary:** its probabilities are not meaningful → block could mislead;
   keep the flag off for mock runs (document).
5. **Minor cost/length:** ~50–100 extra prompt tokens per escalated call.

## Suggested implementation sequence (when approved)
1. `render_primary_block` helper + top-2 util (+ tests).
2. Optional `$primary_block` placeholder + `primary_signal` param in the 3 prompts
   (default empty → backward compatible).
3. Config flag (`TaskConfig` + loader + default.yaml), default **off**.
4. Agents build/pass `primary_signal` when the flag is on.
5. Tests (above); full offline suite stays green (flag off = no change).
6. **Validation:** offline confirms empty-when-off and correct render-when-on; then
   a **paid A/B** on EESA full_agentic (flag off vs on, with Fix #2 active) —
   separate approval — reporting accuracy/macro-F1 *and* agent-vs-primary agreement
   rate to detect anchoring. Consider a small joint check with a lower w_primary.
