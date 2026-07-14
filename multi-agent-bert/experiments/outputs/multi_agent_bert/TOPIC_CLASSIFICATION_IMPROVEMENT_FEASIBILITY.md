# Topic Classification — Improvement Feasibility Audit

Strict cost-benefit analysis of whether any change to the ARENTCV2 topic path is scientifically
justified. Uses existing artifacts only — no new runs. Date: 2026-07-07.

## Executive summary
**Recommendation: A — no further improvement is justified; use primary_only as the topic system.**
The decisive fact: of the **112** primary errors on 21,134, **95 (85%) are HIGH-confidence errors
that never escalate** — the agents cannot see them. Agents can touch only the **17** escalated
errors (15%), of which ~82% are **arbitrary/inconsistent gold** (established in the gold-quality
analysis). Realistic recoverable improvement is **~2–3 samples out of 21,134 (≈ +0.0001)** — deep
inside noise. Per the stated stopping rule, **stop.**

## 1. Current topic performance
| system | accuracy | macro F1 |
|---|---|---|
| primary_only | **0.9947** | 0.9947 |
| full_agentic (default agents) | 0.9944 | 0.9944 |
| full_agentic (reason-first + sharpened) | 0.9947 | 0.9947 |

Agents are **neutral** on the full set (match or slightly below the primary).

## 2. Remaining error analysis (primary_only)
- **112 total errors** on 21,134 (acc 0.9947).
- **Error confidence is extreme:** median error confidence = **0.998**; **87** errors have conf ≥
  0.95; **73** have conf ≥ 0.99. The model is *confidently* wrong.
- **Confusion pairs (all 112):**
  | pair | count | share |
  |---|---|---|
  | **health ↔ medical** | **42** | **37.5%** |
  | tech ↔ finance | 20 | 18% |
  | tech ↔ business | 14 | 12.5% |
  | finance ↔ business | 13 | 12% |
  | health/social/sports misc | ~23 | 20% |
- The dominant confusion (health↔medical, 42) is the **arbitrary-gold** pair — the gold-quality
  analysis proved the dataset labels telemedicine/clinical concepts inconsistently (health 68 /
  medical 120 for "telemedicine"). So a large share of these "errors" are **label errors, not model
  errors** — not fixable by any agent.
- **Truly ambiguous / fixable?** Mostly ambiguous (overlapping labels) + inconsistent gold; a
  minority are clean errors, and those are **high-confidence**, so agents never reach them.

## 3. Escalated subset analysis (threshold 0.90)
- **Escalated: 48 / 21,134 (0.23%).** Of these, **17 are primary errors**, 31 already correct.
- **Agents can only ever touch 17 of the 112 total errors (15%).** The other **95 errors are
  high-confidence and never escalate.**
- Accuracy on the 48 escalated: primary **31/48 (0.646)**, reason-first agents **31/48 (0.646)** —
  **net 0 (W→C 11, C→W 11).** Agents are **NOT better** than the primary on the escalated topic
  cases; they reason to a tie, and (per gold-quality) are more taxonomy-consistent than the gold
  but score the same on the noisy labels.

## 4. Agent behavior analysis
- **Net:** agents changed 22/48 escalated decisions (W→C 11, C→W 11) → no net gain.
- **Per-agent diversity / strongest agent / which agent breaks predictions: NOT MEASURABLE** from
  saved artifacts — the topic runs stored only final labels, no per-agent votes. Determining this
  would need a per-agent capture (~$0.04). **Given §7 (max ≈ +2–3 samples), that capture is not
  cost-justified** — it cannot change the recommendation.
- What *is* known: no agent is provably useful here because the panel nets to zero on the only
  subset it sees.

## 5. Prompt improvement assessment
- Prompts are **already task-generic and correct** (audit: all 9 labels, no sentiment/EESA, no
  hardcoded labels).
- **Already tested and exhausted:** sharpened label descriptions (tech rule 1/7→7/7) and reason-first
  prompting took the escalated agents **25 → 28 → 31 = primary**. Further prompt work is capped by
  the arbitrary-gold half (health/medical stuck 12–13/25 regardless).
- **Adding examples → overfitting risk** on inconsistent gold (would fit label noise, not signal).
- **Reason-first beyond current results: no headroom** — it already reached parity; the residual is
  data, not prompting.

## 6. New-agent assessment (conceptual only)
Evaluated against the error data; **none justified:**
- **Health-vs-medical disambiguation agent:** the pair is the biggest error source (42), *but* ~40
  of those are **non-escalated high-confidence** (agent never sees them) and the escalated ones are
  **arbitrary gold**. A specialist would reach ~11 escalated cases, most unfixable. **Not justified.**
- **Business-vs-finance / tech-vs-finance / education-vs-tech agents:** same structure — the errors
  are mostly non-escalated and/or ambiguous; a specialist would touch a handful of escalated cases
  at best. **Not justified.**
- **General rule met?** No: the confusion is repeated but **not fixable** (unreachable + arbitrary
  gold), so the "repeated, fixable confusion" bar is **not** cleared.

## 7. Routing / consensus assessment
- **Lower threshold (escalate more):** floods the agents with confident-correct predictions they
  break (Flaw 1) → net harmful. **No.**
- **Raise threshold (escalate less):** → converges to primary_only. **No gain.**
- **Margin-based routing (Flaw 1 fix):** even this cannot recover the 95 confident errors — they
  have median conf 0.998 *and* therefore large top-2 margins, so they look "easy" to any
  confidence/margin router. The errors are undetectable from the primary's own signal. **No.**
- **IntentGate-style guard for topic: unnecessary** — the gate protects a *neutral sentiment*
  primary from polar overrides; topic has no neutral class and no opinion/mention axis. It would be
  a meaningless import. **No.**
- **Consensus changes:** the panel already nets zero on the reachable subset; re-weighting cannot
  create signal that isn't there. **No.**

## 8. Cost-benefit analysis
- **Maximum theoretical improvement:** fix all 17 reachable escalated errors → 0.9947 → **0.9955**
  (+17 samples). Impossible in practice (agents already tie the primary there; many are unrecoverable
  or arbitrary gold).
- **Realistic recoverable:** ~**2–3 samples** out of 21,134 → **≈ +0.0001**, well inside temp-0 /
  seed noise.
- **The 95 high-confidence errors are structurally unreachable** by the escalation paradigm, and
  partly are gold errors themselves.
- **Cost of chasing it:** new agents / captures / prompt cycles for **≤ 3 samples** — not justified.

## 9. Final recommendation — **A**
**A. No further improvement needed; use primary_only for the thesis.**

The topic primary is at ceiling (0.9947); 85% of its errors are high-confidence and unreachable by
agents, and the reachable 15% is dominated by arbitrary/inconsistent gold. Prompt improvements were
already tried and reached parity; new agents, routing changes, and a topic gate are all ruled out by
the error structure. Realistic upside is 2–3 samples — below the stopping threshold.

**Report full_agentic as a validation experiment (the spirit of option E), not the final system:**
it demonstrates the multi-agent layer is **neutral (does not hurt)** on a near-perfect primary and,
via the gold-quality finding, is **more taxonomy-consistent than the labels** — a defensible result.
But the **deliverable topic system is primary_only.**

The only genuine improvement lever is **data-side** (de-noising the health↔medical / fintech gold),
which is out of scope for the model/agent pipeline.

## Artifacts
- `experiment_T2_reasoned_full/` (primary + full predictions), `TOPIC_CLASSIFICATION_PIPELINE_AUDIT.md`,
  `EXPERIMENT_T2_GOLD_QUALITY_ANALYSIS.md`, `EXPERIMENT_PIPELINE_FLAW_ANALYSIS.md`.
