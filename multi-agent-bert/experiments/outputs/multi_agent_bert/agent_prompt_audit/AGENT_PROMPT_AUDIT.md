# Agent / Prompt / Router / Consensus Audit

Inspection-only design review of the `full_agentic` path. No code changed, no
experiments, no API calls. Date: 2026-06-09.

Scope reviewed: `src/agents/`, `src/prompts/`, `src/pipeline/`, `src/config/`,
`evaluate_pipeline.py` wiring.

---

## 1. Prompt generality

| Prompt (system) | Generic? | Task/label wording found |
|---|---|---|
| [llm_lexical_prompt.py](../../../src/prompts/llm_lexical_prompt.py) `SYSTEM_PROMPT` | **No** | "identify the most likely **topic** label", "what **domain** this text … belongs to" |
| [llm_logic_prompt.py](../../../src/prompts/llm_logic_prompt.py) `SYSTEM_PROMPT` | **No** | "determine the most likely **topic** label", "point to one **domain**" |
| [contextual_prompt.py](../../../src/prompts/contextual_prompt.py) `SYSTEM_PROMPT` | **Yes** | "strict text classification engine" — neutral |
| [llm_explainability_prompt.py](../../../src/prompts/llm_explainability_prompt.py) | **Yes** | task-agnostic |
| [deliberation_prompt.py](../../../src/prompts/deliberation_prompt.py) | **Yes** | task-agnostic |
| ConsensusAgent | n/a — **no prompt**, pure weighted vote in code | — |

- **No prompt hardcodes `positive`/`negative`/`neutral`.** Labels and
  descriptions are always injected from config (`$labels_csv`, `$labels_block`,
  `$task_name`). That part is already task-config driven.
- **But the lexical and logic system prompts hardwire a `topic`/`domain`
  frame.** They literally instruct the model to pick a "topic label" / decide a
  "domain" — semantically wrong for sentiment. They only "work" today because
  the injected `label_descriptions` carry the real sentiment meaning, so a
  capable model overrides the bad framing.
- **Would the same prompt serve sentiment *and* topic without a rewrite?**
  Mechanically yes (labels come from config); semantically the lexical/logic
  prompts are topic-biased, so they mis-frame sentiment now and merely happen to
  match topic later. Contextual/explainability/deliberation prompts are already
  generic.

## 2. Config usage per agent

| Signal | LLMLexical | LLMLogic | Contextual | LLMExplain | Deliberation | Consensus | Router |
|---|---|---|---|---|---|---|---|
| `task_config.labels` | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ (validate) |
| `label_descriptions` | ✅ | ✅ | ✅ | — | ✅ | ❌ | ❌ |
| `task_name` / active_task | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `keyword_map` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `rule_map` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| primary **prediction** | ❌ | ❌ | ⚠️ only via opt-in summary | ✅ (final label) | ⚠️ via votes | ❌ (not a vote) | ✅ |
| primary **confidence** | ❌ | ❌ | ❌ | ✅ (final conf) | ❌ | ❌ | ✅ |
| primary **full probabilities** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **top-2 candidate labels** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Notes:
- `keyword_map`/`rule_map` are consumed only by the **non-LLM** `LexicalAgent`/
  `LogicAgent` (paper_style). In `full_agentic` they are dead wiring.
- Contextual can receive prior-agent summaries, but only if
  `execution.contextual_use_prior_outputs` is true — it is **false** in
  [default.yaml](../../../src/config/default.yaml), so in the real-LLM pilot the
  agents were **blind to the primary**. Even when on, summaries are free-text
  "weak hints" and exclude primary probabilities / top-2.
- **No agent ever sees the primary's probability distribution or its top-2
  labels.** Each LLM agent re-classifies from scratch.

## 3. Fallback behavior — first-label bias is real

`labels[0]` is the fallback in **six** places (grep-verified):

| File:line | When |
|---|---|
| [consensus_agent.py:207](../../../src/agents/consensus_agent.py#L207) | no usable votes |
| [contextual_agent.py:306](../../../src/agents/contextual_agent.py#L306) | parse failure |
| [llm_lexical_agent.py:209](../../../src/agents/llm_lexical_agent.py#L209) | parse / invalid-label |
| [llm_logic_agent.py:208](../../../src/agents/llm_logic_agent.py#L208) | parse / invalid-label |
| [lexical_agent.py:109](../../../src/agents/lexical_agent.py#L109) | no keyword hit |
| [logic_agent.py:118](../../../src/agents/logic_agent.py#L118) | no rule hit |

- For sentiment, `labels = [positive, negative, neutral]` → every fallback emits
  **positive**. This is precisely the **positive-collapse mechanism** the mock
  full_agentic and the threshold sweep exposed (negative W→C was 0 in every
  cell). A degenerate agent that always parse-fails becomes a positive emitter.
- A **no-vote / abstain** fallback would be safer: a failed agent should
  contribute *no* vote, and an all-failed consensus should defer to the
  **primary**, not to `labels[0]`.

## 4. Consensus behavior

[consensus_agent.py](../../../src/agents/consensus_agent.py):
- **Primary is NOT a vote.** `agent_slots` = lexical, contextual, logic (+
  optional deliberation) only. On escalation the router does **not** set
  `final_output` (only the `accept_primary` branch does), so the final label is
  **entirely** the agents' — *the primary prediction is discarded.*
- Aggregation = **confidence-weighted vote**: `score[label] += weight *
  agent_confidence`, all weights default `1.0`. Final confidence = winning score
  / active weight sum.
- **Ties** break by earliest label in `task_config.labels`
  (`max(..., key=lambda l: (scores[l], -labels.index(l)))`, line 234) → **first-
  label (positive) bias** again.
- **LLM self-reported confidences are treated as calibrated** and used directly
  as vote weights — a confidently-wrong agent dominates.
- **No conservative override.** Three weak agents (each weight 1.0) outvote a
  primary that isn't even on the ballot. Agents can override the primary with
  zero resistance.

## 5. Router behavior

[router.py:33](../../../src/pipeline/router.py#L33):
- Escalation signal is **only** `primary.confidence >= threshold`. No
  probability **margin**, no **entropy**, no **top-2 closeness**.
- Threshold is **global**: it comes from `execution.threshold`
  ([loader.py:145](../../../src/config/loader.py#L145)), a single value for
  whichever `active_task` — **not** per-task. (Defaults are also scattered: 0.7,
  0.65, 0.5 across loader/task_config.)
- **Topic generalization risk:** with 9 classes the max softmax prob is
  typically lower than with 3, so a fixed 0.6 threshold will over-escalate on
  topic and route far more (more cost, and more chances for weak agents to
  override). A confidence-only, single-global threshold will not transfer
  cleanly.

## 6. Why the real-LLM pilot improved — and what's still fragile

**What drove the gain (real GPT-4o-mini vs mock):**
- The **Contextual agent** has the only generic, correct prompt; with a real
  model it produced genuine semantic judgments on the low-confidence escalated
  slice, reading the injected sentiment `label_descriptions`.
- **JSON mode** eliminated the mock's parse failures (0 parse warnings), so
  agents stopped silently collapsing to the `labels[0]` fallback.
- Three *real* votes (even lexical/logic, mis-framed) mostly agreed on the
  correct minority label, so confidence-weighted voting landed right — and the
  **negative** class recovered.

**Still fragile (these did not bite only because GPT-4o-mini is competent):**
- Primary is discarded on escalation → no safety net when agents are wrong.
  This is benign at threshold 0.6 (few, genuinely-hard samples) but scales
  badly as the threshold rises or on harder tasks.
- `labels[0]`/positive fallback + positional tie-break remain latent positive
  bias, ready to re-emerge on any API error, refusal, or ambiguous tie.
- Lexical/logic "topic/domain" framing is a lucky mismatch for sentiment.
- Uncalibrated LLM confidences as raw weights.

**Fix before scaling to topic classification:** generic prompts; primary-aware
conservative consensus; abstain (not `labels[0]`) fallbacks; per-task threshold;
give agents the primary's top-2/probabilities.

---

## 7. Ranked flaws

### CRITICAL

**C1 — Consensus discards the primary; agents override with no resistance.**
- *Where:* `consensus_agent.py::ConsensusAgent.run` (no primary slot) +
  `router.py` (escalate path sets no `final_output`).
- *Why:* the fine-tuned transformer (the strongest single component, ~0.80–0.82)
  is thrown away the moment a sample escalates; the label comes only from three
  equally-weighted agents.
- *Evidence:* `agent_slots = {lexical, contextual, logic}` (line 144); router
  sets `final_output` only under `accept_primary` (line 36).
- *Sentiment impact:* caused the mock regressions and the monotonic decline in
  the threshold sweep (net −68 at 0.9). With real agents it's currently positive
  but unprotected.
- *Topic impact:* worse — 9-class agents are likelier to be wrong; discarding a
  strong primary will regress more often.
- *Generic fix:* make the consensus **primary-aware** via config: add the
  primary as a weighted vote (`weights.primary`, task-config driven), or a
  conservative override — keep the primary label unless the agents agree
  *and* their combined score beats the primary confidence by a configurable
  margin. No label hardcoding.

**C2 — First-label (`labels[0]`) fallback = positive bias.**
- *Where:* 6 sites (table in §3), incl. consensus no-vote and every LLM-agent
  parse/invalid-label path.
- *Why:* any failure emits the first configured label; with sentiment that is
  always `positive`.
- *Evidence:* grep of `labels[0]` / `fallback_label`.
- *Sentiment impact:* systematic positive inflation on errors/ties; the exact
  failure mode seen with the mock.
- *Topic impact:* would bias toward `business` (first topic label).
- *Generic fix:* **abstain semantics** — a failed agent returns *no vote*
  (excluded from consensus); if consensus ends with zero votes, **defer to the
  primary prediction**, never to `labels[0]`. Task-agnostic.

### MEDIUM

**M1 — Lexical/logic system prompts hardcode "topic"/"domain".**
- *Where:* `llm_lexical_prompt.py`, `llm_logic_prompt.py` `SYSTEM_PROMPT`.
- *Evidence:* "most likely topic label", "what domain this text belongs to".
- *Sentiment impact:* mis-frames the task; relies on label descriptions to
  rescue it. *Topic impact:* accidentally correct, still not generic.
- *Fix:* neutral framing — "classify the text into exactly one of the allowed
  labels for task `$task_name`", semantics carried only by `label_descriptions`.

**M2 — Positional tie-break favors the first label.**
- *Where:* `consensus_agent.py:234`. *Evidence:* `-labels.index(lbl)`.
- *Impact:* ties → positive (sentiment) / business (topic).
- *Fix:* break ties toward the **primary's** label; if primary not in the tie,
  lower confidence / abstain. Never positional.

**M3 — LLM self-reported confidences treated as calibrated weights.**
- *Where:* `consensus_agent.py` `score += weight * confidence`.
- *Impact:* a confident-but-wrong agent dominates; uncalibrated across agents
  and tasks.
- *Fix:* treat agent confidence as ordinal — clamp/bucket it, or weight by
  inter-agent agreement rather than raw self-confidence (config-driven).

**M4 — Router: confidence-only signal + single global threshold.**
- *Where:* `router.py:33`, `loader.py:145` (global `execution.threshold`).
- *Impact:* no margin/entropy/top-2; a 3-class-tuned 0.6 will over-escalate on
  9-class topic.
- *Fix:* per-task threshold (`tasks.<name>.threshold`, fallback to global) and
  an optional margin/entropy escalation signal; keep confidence as default.

**M5 — Agents are blind to the primary (no top-2 / probabilities passed).**
- *Where:* LLM agent prompts receive only labels+descriptions+text; contextual
  prior-summaries default off and exclude probabilities.
- *Impact:* agents re-classify from scratch and cannot focus on the primary's
  actual top-2 confusion, nor act conservatively relative to it.
- *Fix:* inject the primary's **top-2 labels + probabilities** as structured
  prompt fields (generic across tasks); let agents confirm/deny the primary
  rather than guess blind.

### MINOR

- **m1** `keyword_map`/`rule_map` unused in full_agentic — dead wiring (paper_style only).
- **m2** LLM explainability runs per escalated sample (~25% of pilot cost) but is post-hoc and never changes the label — could be made optional / template for cost.
- **m3** Deliberation, even when enabled in `build_orchestrator`, is wired to a `MockLLMClient(fixed)` — it cannot use the real OpenAI client today (off by default, but a future gap).
- **m4** Scattered threshold defaults (0.7 / 0.65 / 0.5 across loader/task_config) — consolidate to one source of truth.

---

## Design direction (matches the stated preference — no sentiment hardcoding)

All fixes stay task-config driven; **none** special-case positive/negative/neutral:
1. Generic prompts (drop "topic"/"domain"); semantics only from `label_descriptions`.
2. Abstain (no-vote) fallbacks; consensus defers to **primary**, never `labels[0]`.
3. Primary-aware **conservative consensus** (primary as a weighted vote or
   margin-gated override) + non-positional tie-break.
4. Per-task threshold in config; optional margin/entropy router signal.
5. Pass the primary's top-2 labels + probabilities into agent prompts.
6. Treat LLM confidences as ordinal, not calibrated probabilities.

These generalize unchanged to topic classification (9 labels) and any future
task, because every task-specific value continues to come from `task_config`.
