# Sum-Rule vs Hard-Vote Fusion — Ahmed G2 (gpt-4.1-mini)

Does replacing the pipeline's confidence-weighted **hard voting** with the
literature-standard **sum rule** (Kittler et al.) change results? Opt-in
`ConsensusAgent(fusion="sum_rule")`; default `hard_vote` unchanged. One paid
pass over the Ahmed frozen-primary G2 selective-gate setup (semantic_v1,
threshold 0.70, w_primary 1.0, gpt-4.1-mini); the 84 escalated samples were
re-fused OFFLINE under both schemes by the production `ConsensusAgent` +
`IntentGateAgent` (no second LLM call). Date: 2026-07-19.

## What differs between the two schemes
- **hard_vote** (legacy): each voter adds `weight × confidence` to its argmax
  label only; residual mass `(1 − confidence)` is discarded. Scores do not sum
  to the voter count, so the reported number is a **weighted vote share**.
- **sum_rule**: each voter adds `weight × P(label)` across *all* labels, using
  its full distribution — the primary's real softmax, the agents' uniform
  spread (`_build_probabilities`). Scores sum to `Σ weights`, so the reported
  number is a **proper posterior**. Label count is read from config (task-agnostic).

## Headline

| metric | hard_vote | sum_rule |
|---|---|---|
| **accuracy (full 818)** | **0.9303** | **0.9303** |
| accuracy (escalated 84) | 0.7976 (67/84) | 0.7976 (67/84) |
| **label flips (all 818)** | — | **0** |

**Identical accuracy and macro behaviour.** Zero labels change. The `hard_vote`
path also reproduces the prior best G2 number (0.9303) exactly, validating the
refactor. → Adopting the sum rule would **not** alter any accuracy/F1 in the
thesis tables.

## Calibration (escalated subset — where fusion sets the confidence)

| metric | hard_vote | sum_rule | note |
|---|---|---|---|
| **ECE** (lower better) | 0.1220 | **0.0982** | −19% relative — better calibrated |
| mean conf \| correct | 0.6939 | 0.7138 | closer to escalated acc 0.798 |
| mean conf \| wrong | 0.6036 | 0.6429 | also rises |
| ECE (full 818) | 0.1389 | 0.1365 | marginal |

Both schemes are **under-confident** on the escalated subset (mean-conf-correct
< accuracy 0.798). The sum rule lifts confidences toward the true accuracy, so
ECE improves. **Caveat (honest):** the correct-vs-wrong *separation* does not
improve — the gap narrows slightly (0.090 → 0.071) because wrong-case confidence
also rises. So the sum rule is **better calibrated, not better at discriminating**.

## Gate-fired cases (the low-confidence pathology)
The IntentGate reverts an override to the primary; the reported confidence then
comes from the primary's lone vote, which is very low under hard-vote. The sum
rule mitigates but does not fully fix this (still low, just less extreme):

| sample | true | final | hard | sum |
|---|---|---|---|---|
| ahmed-eesa-00097 | negative | negative ✓ | 0.097 | 0.185 |
| ahmed-eesa-00203 | neutral | neutral ✓ | 0.142 | 0.204 |
| ahmed-eesa-00193 | negative | negative ✓ | 0.156 | 0.231 |
| ahmed-eesa-00330 | neutral | neutral ✓ | 0.381 | 0.425 |

## Conclusion
The sum rule is the **principled** combiner (proper posterior, keeps the
primary's second-place mass) and is **strictly safe** to adopt: identical
labels, identical accuracy/F1, better calibration (ECE −19% on escalated). It
does not improve discrimination. Recommendation: keep `hard_vote` as the default
for continuity with all prior tables; cite `sum_rule` as an implemented,
verified variant showing our reported metrics are unaffected by the fusion choice.

Artifacts: `experiment_ahmed_sumrule_vs_hardvote/{refusion_records.json,sumrule_vs_hardvote_report.txt}`.
Code: `ConsensusAgent(fusion=...)` in `src/agents/consensus_agent.py`; tests in
`tests/test_consensus_agent.py::TestSumRuleFusion`.
