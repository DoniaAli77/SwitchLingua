# Fix-Order Memo (classification flow)

Comparison of the four remaining fixes, to decide sequencing. **No
implementation.** All target the shared classification flow (sentiment + topic);
NER frozen. Date: 2026-06-10.

Audit cross-refs: C1 = primary discarded by consensus, C2 = `labels[0]` fallback,
M5 = agents blind to primary, M4 = router.

---

## 1. No-vote (abstain) fallback instead of `labels[0]`  — audit C2

| | |
|---|---|
| **Code changes** | In the LLM agents (`llm_lexical_agent`, `llm_logic_agent`, `contextual_agent`) make the parse/invalid-label fallback emit **no vote** (label `None`) instead of `labels[0]`. In `consensus_agent`, when no usable votes remain, **defer to the primary prediction** (`state.primary_model_output.label`) instead of `labels[0]`. The vote extractor already drops `None`-label outputs, so abstain is mostly supported already. |
| **Task-generic?** | **Yes** — purely positional-bias removal; no label names involved. |
| **Impact — sentiment** | Removes the latent **positive** bias on any parse error / all-abstain tie. Small under real GPT-4o-mini (0 parse errors today), but eliminates the exact mechanism behind the mock's positive collapse and makes results robust to API hiccups (e.g. the connection-error runs). |
| **Impact — topic** | Larger — removes `business` (first label) bias; first-label bias is more harmful with 9 classes. |
| **Risk** | **Low.** Need to confirm `label=None` flows cleanly (validation, evaluator) and update any test asserting `fallback == labels[0]`. No re-eval needed to be *safe*; behavior only changes on failures/ties. |
| **How to test** | Offline unit tests: a parse-failed agent contributes no vote; consensus with all agents abstaining returns the **primary** label (not `labels[0]`); ties no longer resolve to first label once tie-break is addressed (M2, separate). No API. |

## 2. Primary-aware consensus  — audit C1 (highest-impact safety fix)

| | |
|---|---|
| **Code changes** | `consensus_agent`: add the **primary** as a participant — either (a) a weighted vote slot (`weights["primary"]`, config-driven) using `state.primary_model_output` (label + confidence/probabilities), or (b) a **margin-gated override**: keep the primary label unless the agents' combined score beats the primary by a configurable margin. Wire the weight/margin from `default.yaml` via `build_orchestrator`. No new state field (primary output already in state). |
| **Task-generic?** | **Yes.** |
| **Impact — sentiment** | The big robustness win. The threshold sweep showed the primary often beats weak agents on escalated samples; giving it a vote/veto prevents C→W regressions. With real agents it slightly tempers the upside but cuts the downside — most valuable at higher thresholds where more (harder) samples escalate. |
| **Impact — topic** | High — 9-class agents are weaker and likelier wrong; a primary safety vote is the main guard against agent-driven collapse. |
| **Risk** | **Medium.** Changes the final label for **every** escalated sample, so it needs a **paid real-LLM re-eval** to tune the weight/margin; a mis-set weight could erase the pilot's gains (too much primary) or under-protect (too little). Interacts with fix #1 (no-vote→primary is its degenerate case). |
| **How to test** | Offline unit tests for the new vote math / override logic. Effect validation = one real-LLM full_agentic re-run on EESA at 0.6 (and maybe 0.8) — small cost, after the sweep. |

## 3. Optional primary-signal block in agent prompts  — audit M5 / proposal §E

| | |
|---|---|
| **Code changes** | Prompt templates (lexical/logic/contextual) gain an optional `$primary_block`; the three agents read `state.primary_model_output` (label, confidence, probabilities), compute **top-2**, and pass `primary_signal` to `build_user_prompt`; a config flag (`execution.agents_use_primary_signal`) gates it (loader + agents). No new state field. |
| **Task-generic?** | **Yes** (label/probabilities come from the config-driven primary). |
| **Impact — sentiment** | Turns agents into **adjudicators** ("primary is torn between negative 0.43 / neutral 0.39") — can lift escalated accuracy further. Counter-risk: **anchoring** (agents echo the primary), which would shrink the ensemble's independent value. |
| **Impact — topic** | Potentially large — agents see the primary's top-2 among 9 classes and can focus the decision; same anchoring risk. |
| **Risk** | **Medium.** Behavior change to agent *inputs*; anchoring could reduce diversity; widest code surface (prompts + 3 agents + config). Most meaningful with the **real** transformer primary (genuine probabilities); keep off for mock. |
| **How to test** | Offline unit: block renders correctly, top-2 computed, flag gates default-off. Effect = paid **A/B** (block on vs off) on EESA — do only after #1/#2 settle. |

## 4. Task-specific / margin-based router  — audit M4

| | |
|---|---|
| **Code changes** | `router.py`: optional **margin** (top1−top2) or **entropy** escalation signal alongside confidence. `loader`/`task_config`/`default.yaml`: **per-task threshold** (`tasks.<name>.threshold`) overriding the single global `execution.threshold`. Config-schema change. |
| **Task-generic?** | **Yes** (per-task config is the generalization mechanism). |
| **Impact — sentiment** | Near-neutral — 3-class with 0.6 already works; per-task threshold just lets sentiment keep its tuned value. Margin/entropy is a minor refinement here. |
| **Impact — topic** | **Important** — a 3-class-tuned 0.6 over-escalates on 9-class (lower max-softmax); topic needs its own threshold and benefits most from margin/entropy. **But topic data isn't ready yet**, so the payoff can't be validated now. |
| **Risk** | **Low–Medium.** Config-schema change; for sentiment it's near-neutral, so low risk to current results. Can't validate the topic benefit until topic data lands. |
| **How to test** | Offline unit (per-task threshold resolution, margin/entropy computation). For sentiment, **no new API needed** — re-derive thresholds from existing saved predictions (as in the sweep). Topic validation deferred to when topic data exists. |

---

## Recommended order (with rationale)

1. **No-vote fallback (C2)** — *first*. Lowest risk, fully task-generic, removes a
   clear bias, almost entirely offline-testable, and is the degenerate case of
   primary-aware consensus (sets up #2 cleanly).
2. **Primary-aware consensus (C1)** — *second*. Highest-impact safety fix (the top
   critical). Build on #1's "defer to primary" notion; tune weight/margin with one
   small paid re-eval.
3. **Primary-signal block (M5)** — *third*. An enhancement on top of a now-safe
   consensus; A/B it for anchoring before trusting it.
4. **Task/margin router (M4)** — *last (or when topic data lands)*. Mostly a
   **topic** enabler; near-neutral for sentiment and not validatable on topic yet.
   Prep the per-task-threshold plumbing whenever convenient, but it's the lowest
   urgency for the current sentiment work.

Sentiment-now priority: **1 → 2** are the meaningful ones; **3** is an upside
experiment; **4** is mostly for the topic phase. Each is its own small PR with
offline tests; #2 and #3 additionally need a small paid EESA re-eval to confirm
effect. **No implementation until you pick the order.**
