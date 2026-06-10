# Design Proposal — Primary-Aware Consensus (audit C1)

**Proposal only — nothing implemented, no OpenAI calls.** Makes `ConsensusAgent`
include the primary model as a weighted vote/prior on the escalation path instead
of discarding it. Task-generic; router unchanged; tie-break de-positionalized;
no primary-signal prompt block; LLM prompts untouched. Date: 2026-06-11.

## 1–2. How consensus currently combines votes
[consensus_agent.py:138-267](../../../src/agents/consensus_agent.py#L138).
For each of `lexical / contextual / logic` (and optional `deliberation`) that did
**not** abstain:
```
scores[label] += weight[slot] * agent_confidence
active_weight_sum += weight[slot]
```
Winner = `max(labels, key=lambda l: (scores[l], -labels.index(l)))` — i.e. highest
weighted score, ties broken by **earliest label in config order** (positional).
`final_confidence = scores[winner] / active_weight_sum`. Default weights:
lexical/contextual/logic = 1.0, deliberation = 0.0.

## 3. Where the primary is discarded
On the **some-votes** path (lines 249-267) the primary is **never** added to
`scores`. The router does not set `final_output` on escalation either, so once a
sample escalates and ≥1 agent votes, the final label is **agents-only** — the
fine-tuned primary (often the strongest single component) is thrown away. (Fix #1
added the primary only in the *all-abstain* corner.)

## 4. Proposed: primary as a weighted vote (prior)
Add the primary to the **same** `scores` accumulator, before choosing the winner:
```
primary = state.primary_model_output
if primary.label is not None and task.is_allowed_label(primary.label):
    scores[primary.label] += w_primary * primary.confidence
    active_weight_sum     += w_primary
```
- The primary becomes one participant among the agents — a **prior** anchored on
  its own (softmax) confidence. Argmax then naturally implements a conservative
  override: agents flip the primary only when an alternative label's combined
  agent score **exceeds** the primary's contribution.
- This **unifies Fix #1**: if every agent abstains, only the primary votes → the
  primary wins automatically. The old "all-abstain" branch collapses to a single
  guard: *no votes at all (agents abstained AND no usable primary)* → no-decision
  (`label=None`), never `labels[0]`.

## 5. Weighting strategy
| Knob | Proposal |
|---|---|
| **primary weight `w_primary`** | New `execution.consensus_weights.primary`, config-driven. Default a **modestly conservative** value (start at **1.0**, validate over {1.0, 1.5, 2.0}); higher = more conservative. `w_primary = 0` exactly reproduces today's agents-only behavior (ablation / backward-compat). |
| **lexical/logic/contextual** | Unchanged (1.0 each), still config-driven. |
| **abstentions** | Already excluded by `_extract_vote` (Fix #1). Abstainers contribute nothing and don't dilute `active_weight_sum`. |
| **confidence values** | Primary vote = `w_primary * primary.confidence` (softmax confidence is reasonably calibrated, so a near-threshold-confident primary anchors *more* than a barely-confident one — directly handles case 8.4). Agent votes keep `weight * agent_confidence` **unchanged** (their self-confidences are uncalibrated — that's audit M3, a *separate* fix). **Optional** guard: clamp agent confidence to `[0, c_max]` so one overconfident agent can't single-handedly flip the primary; off by default, flagged for M3. |

Why scale the primary by its confidence (not a flat weight): on a **high**
threshold we escalate confident-but-below-cutoff primaries (e.g. conf 0.85 at
threshold 0.9) — those should be hard to override; a low-confidence escalated
primary (conf 0.45) should be easier. Confidence-scaling gives both for free.

## 6. When agents may override the primary
Override happens **iff** some alternative label `a ≠ primary.label` satisfies
`scores[a] > scores[primary.label]` (plain argmax once the primary vote is in).
Concretely, agents must muster combined weighted confidence for a *single*
alternative exceeding `w_primary * primary.confidence` (+ any agent votes that
*agree* with the primary). Tuning `w_primary` sets how much agent consensus is
required:
- `w_primary ≈ 1.0`: ~one confident agent's worth — agents override readily.
- `w_primary ≈ 2.0`: needs a strong agent majority to flip — conservative.

**Optional hysteresis** `δ` (config `consensus_override_margin`, default 0):
require `scores[a] - scores[primary.label] ≥ δ` to flip, avoiding overrides on
near-ties. Recommend keeping `δ=0` initially and tuning only if needed.

## 7. Tie-breaking without label order
Replace `-labels.index(l)` with a deterministic, **config-order-free** rule.
Let `T` = labels tied for the max score:
1. If `primary.label ∈ T` → choose **primary.label** (conservative anchor).
2. Else → the label in `T` backed by the **most contributing agents** (count).
3. Still tied → the label in `T` with the **highest single agent contribution**.
4. Astronomically-rare residual tie → deterministic **sorted-by-name** (an
   arbitrary-but-stable rule that is *not* the config's label order).

This never consults `task_config.labels` ordering, so it removes the first-label
(positive / business) bias (audit M2 is effectively absorbed here for the
consensus winner; the per-agent M2 sites were already handled by Fix #1's
abstain).

## 8. Required behaviors
| Case | Behavior |
|---|---|
| **All agents abstain** | Only the primary votes → **primary wins** (subsumes Fix #1). If no usable primary either → no-decision (`label=None`), never `labels[0]`. |
| **Agents disagree (split)** | Weighted argmax with the primary as an anchor vote; a split agent field can't flip the primary unless one alternative out-scores it. Primary breaks weak splits. |
| **Primary confidence low** (typical escalation) | Primary contributes `w_primary * (low conf)` → small anchor → agents can override with modest consensus (appropriate: a low-confidence primary is less trustworthy). |
| **Primary confidence high but escalated under a high threshold** | Primary contributes `w_primary * (high conf)` → large anchor → agents need strong consensus to flip (appropriate: don't lightly override a near-threshold-confident primary). |

## 9. Task-generic
No label names anywhere; everything flows from `task_config.labels`,
`state.primary_model_output`, and config weights. Works unchanged for sentiment,
topic (9 labels), or any future classification task.

## 10–12. Out of scope (unchanged)
Router untouched; no primary-signal prompt block; LLM prompts untouched; agent
confidence calibration (M3) deferred; per-agent tie-break already handled by Fix #1.

## Affected files / surface
- `consensus_agent.py`: add `"primary"` to `_DEFAULT_WEIGHTS` + `__init__` slot
  handling; add the primary vote into the score loop; replace the winner/tie-break
  with `_select_winner(scores, primary_label, vote_counts)`; simplify the
  no-votes guard to "no agents *and* no primary → no-decision".
- `src/config/default.yaml`: `execution.consensus_weights.primary` (and optional
  `consensus_override_margin`).
- `evaluate_pipeline.py` `build_orchestrator`: pass the configured consensus
  weights into `ConsensusAgent(weights=...)` (currently constructed with
  defaults) so `w_primary` is wired from config.
- **No change**: router, prompts, schema, NER, the LLM agents.

## 13. Tests needed (all offline, no OpenAI)
- **Primary is counted:** agents + primary agree → that label wins; `votes`
  shows the primary contribution.
- **Conservative override blocked:** one agent disagrees (high conf) vs a
  high-conf primary with `w_primary=1.5` → **primary label kept**.
- **Override allowed:** two agents agree on an alternative with enough combined
  weight → **flips** the primary.
- **Confidence sensitivity:** the same agent votes flip a *low*-confidence
  primary but not a *high*-confidence one.
- **Tie-break is non-positional:** construct a score tie where
  `primary.label != labels[0]` → **primary.label** wins (not `labels[0]`); and a
  tie where the primary isn't in the tie → most-agents rule wins, never `labels[0]`.
- **All abstain + primary → primary** (Fix #1 preserved through the new path).
- **No agents + no primary → `label=None`** (`no_decision`), never `labels[0]`.
- **`w_primary = 0` reproduces** the exact pre-fix agents-only winner (ablation).
- **Task-generic:** repeat a couple with arbitrary labels `["a","b","c","d"]`.
- **Existing consensus tests:** those built via `make_state()` set **no primary**
  (`primary.label is None`) → the primary vote is skipped → they remain valid
  unchanged. Only tests that explicitly set a primary need review.

## 14. Expected impact
- **Sentiment — `full_agentic` real-LLM:** the agents are *good* on the escalated
  slice (sweep: ~0.65-0.73 vs primary ~0.42-0.56), so a conservative primary
  weight **tempers the upside slightly while cutting C→W regressions**. Net
  depends on `w_primary`; at threshold 0.6 (few escalations) the effect is small.
  Needs a **paid EESA validation** to tune `w_primary` (sweep {1.0,1.5,2.0}); pick
  the value that maximizes macro F1 without erasing the negative-class recovery.
- **Sentiment — `paper_style`:** likely a **clear improvement.** paper_style
  agents are weak (escalated accuracy ≈ 0.20), so anchoring on the primary
  (~0.50+ on the escalated subset) should pull paper_style accuracy up. **Free to
  validate offline** (no API) — do this first.
- **Future topic classification:** important protection — 9-class agents are
  likelier wrong, and the primary anchor guards against agent-driven collapse;
  `w_primary` probably wants to be higher for topic. Generic, transfers unchanged.
- **Risk / main tuning tension:** `w_primary` trades "protect against weak agents"
  (high) vs "capture real-LLM agent gains" (low). Too high → drifts back toward
  primary_only; too low → no protection. Mitigation: default modest, **validate
  per setup** (paper_style free; full_agentic one small paid run), keep
  `w_primary=0` as the ablation baseline. Secondary risk: uncalibrated agent
  confidence (M3) can still over-flip — optional `c_max` clamp noted.

## Suggested implementation sequence (when approved)
1. Add `"primary"` weight + config plumbing (default `w_primary` chosen) — wire
   `build_orchestrator` to pass weights.
2. Insert the primary vote into the score loop; simplify the no-votes guard.
3. Replace the winner/tie-break with the non-positional `_select_winner`.
4. Add/extend tests (above); run full offline suite.
5. **Validate `paper_style` offline** (free) before/after; then a **single small
   paid `full_agentic` EESA run** to tune `w_primary` — *separate* from the code
   PR, on your approval.
