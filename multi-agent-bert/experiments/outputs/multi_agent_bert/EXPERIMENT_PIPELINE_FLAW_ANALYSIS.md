# Pipeline Code Review — Structural Flaws (topic-scoped)

Code review of the shared decision path (orchestrator → router → agents → consensus/gate) used by
the **ARENTCV2 topic** pipeline. Flaws are read from the **code itself** (task-agnostic) and
evidenced with **topic data only**. Date: 2026-07-05.

## Scope note
The router / consensus / gate are **one shared codebase** — the same functions run for topic and
for other tasks — so the code-level flaws below are properties of that code. All *evidence* here is
from the ARENTCV2 topic runs; no other task's results are used.

## Verdict
No crash/correctness bugs in the hot path. **Two architectural flaws** (routing selection; fusion
signal) plus three minor issues. On the near-perfect topic primary these flaws are *masked* (only
0.2% escalates), but the code review surfaces them regardless.

---

## FLAW 1 (HIGH, architectural) — the Router escalates by CONFIDENCE, not by likely-wrongness
`router.py`: `decision = "escalate" if primary.confidence < threshold else "accept_primary"`.
The agents only ever see the **low-confidence** subset — but **unsure ≠ wrong.**

**Topic evidence (ARENTCV2, 48 escalated):** the primary was **already correct on 31/48 = 65%** of
what got escalated. The agents changed **22** of the 48 (W→C 11, C→W 11, net 0) — i.e. handed a
subset that is two-thirds already-right, where each change is as likely to break as to fix.

**Why it matters:** even a good agent panel cannot net-help when most of its input was already
correct. Confidence-based routing selects *confusable* samples, not *wrong* ones.

**Fix:** route on a wrongness signal — **top-2 probability margin** (`p1−p2` small), a small
learned error-detector, or primary-vs-cheap-agent disagreement — escalate where the primary is
likely *wrong*, not merely *unsure*.

## FLAW 2 (HIGH, fusion) — weighted voting uses RAW self-reported confidence as the weight
`consensus_agent.py` L232: `contribution = weight * confidence; scores[label] += contribution`.
The vote weight is the agent's **own `confidence`** field. This is a **code fact**: the fusion
trusts an unvalidated self-estimate, so a confidently-wrong agent can outweigh a hesitant-but-right
one. On a well-calibrated agent this is fine; on LLM self-confidence it is risky.

**Topic status:** the *impact* (whether a correct agent was outvoted on topic) is **NOT yet
measured** — the topic runs saved only final labels, no per-agent votes. **This needs a per-agent
capture on the 48 escalated topic cases to confirm** (≈$0.02). I am flagging it as a code-level
risk, not a topic-verified result.

**Fix (if confirmed):** calibrate confidences before weighting, or fuse by agreement/rank rather
than confidence magnitude.

## FLAW 3 (MEDIUM, design smell) — the primary is triple-counted
Read from `consensus_agent.py`: the primary influences the outcome three ways — (1) a
confidence-weighted **vote** in `scores` (`w_primary=1.0`, L319-330), (2) explicit **tie-break
priority** in `_select_winner` (rule 1, L84-86), and (3) the **IntentGate veto** target (L349-357).
Three overlapping anchors to the same label. Code-level observation; makes "consensus" effectively
"primary + adjustments." Exposed knob is `w_primary`, but the tie-break + gate anchoring remain.

## FLAW 4 (LOW) — the IntentGate can only revert TO the primary
`consensus_agent.py` L349-357: `if best_label != primary_label and gate_label == primary_label:
best_label = primary_label`. The gate can only **cancel** an agent override; it can never steer to
its *own* different opinion nor create a label the primary didn't choose. A brake, never a steer.
(Not used on topic by default; code-level note.)

## FLAW 5 (LOW, latent edge, one-line fix) — a confidence-0 vote inflates `active_weight_sum`
L219-234: an agent returning `confidence == 0.0` passes the `label is not None and weight != 0`
guard, adds `weight` to `active_weight_sum` but `0` to `scores` — diluting `final_confidence` and,
in the all-zero case, making every label tie. **Fix:** also skip when `confidence <= 0`.

## FLAW 6 (LOW, inconsistency) — primary vote excluded from tie-break tallies
The primary adds to `scores` (which *creates* ties) but not to `vote_counts`/`max_contribution`
(which *break* them, L319-335 vs L88-97). Rule 1 masks it; cosmetic.

## What is NOT flawed (correct)
Router accept/escalate logic; deterministic non-positional tie-break; abstain-→-defer-to-primary
(never `labels[0]`, L269-312); OOV-label skipping (L224-230); per-stage error fallbacks; the
reason-first prompt change (max_tokens=700 ≫ needed, JSON keys unchanged, order-independent parse).

## Priority (for topic)
1. **Flaw 1 (router selectivity)** — biggest lever; topic-evidenced (65% of escalated already
   correct). Escalate on margin/likely-wrong, not confidence.
2. **Flaw 2 (fusion signal)** — code-level risk; **confirm on topic with a per-agent capture first**,
   then calibrate/rank-fuse if real.
3. Flaw 5 — one-line safety patch.

## Artifacts
- Reviewed: `router.py`, `consensus_agent.py`, `orchestrator.py`, `openai_client.py`.
- Topic evidence: `experiment_T2_reasoned_full/` (48 escalated, primary 31/48, agents net 0).
