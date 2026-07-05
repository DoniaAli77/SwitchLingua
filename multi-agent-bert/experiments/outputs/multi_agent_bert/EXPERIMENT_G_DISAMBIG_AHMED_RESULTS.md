# semantic_v2_disambig on Ahmed (G @ gpt-4.1-mini) — Negative Result

Tests whether the general (non-dataset) disambiguation prompt — replacing the "platform word
→ lean neutral" shortcut with a report/endorse/attack RELATIONSHIP rule + a
description-vs-evaluation rule — fixes the gpt-4.1-mini "wash". Design G, Ahmed frozen
primary, threshold 0.7, gpt-4.1-mini. Date: 2026-07-02.

## Result — it made things WORSE

| system | accuracy | macro F1 | escalated acc | net vs primary |
|---|---|---|---|---|
| primary_only | 0.9254 | 0.9207 | 63/84 = 0.750 | — |
| G @ 4o-mini (semantic_v1) | 0.9279 | 0.9257 | 65/84 = 0.774 | +2 |
| G @ 4.1-mini (semantic_v1) | 0.9291 | 0.9248 | 66/84 = 0.786 | +3 |
| **G @ 4.1-mini (semantic_v2_disambig)** | **0.9242** | 0.9196 | **62/84 = 0.738** | **−1** |

vs G@4o-mini: **fixed 4, broken 8** → net **−4 escalated**. The disambig prompt dropped
accuracy below plain G *and* below primary_only.

## Why it failed — the two-part mechanism
The target cases the rule was written for **did not get fixed**, and the extra instructions
**broke more elsewhere**:

| case | true | what the rule wanted | disambig result |
|---|---|---|---|
| 00240 "عاملين dislike" | neutral | neutral (report) | neutral ✅ (kept) |
| 00298 "الفتوة=dislike" | neutral | neutral (report) | neutral ✅ (kept) |
| **00041** "123 الف dislike يارب تكون الرسالة وصلت" | **negative** | negative (endorse) | **neutral ❌ (NOT fixed)** |
| **00045** "عاملين dislike ليه يا ولاد المرة!" | **negative** | negative (attack) | **neutral ❌ (NOT fixed)** |
| **00362** Breaking Bad plot description | **neutral** | neutral (description) | **negative ❌ (NOT fixed)** |

- **The relationship rule did not resolve the ambiguous cases.** Deciding "is the author
  *endorsing* or *attacking* the dislikes?" is itself a hard pragmatic judgment the model
  still can't make — on 00041/00045 it defaulted to neutral anyway. The rule named the
  distinction but the model couldn't execute it.
- **More instruction added noise.** The longer, rule-heavier prompt degraded decisions on
  *other* cases (8 broken vs 4o-mini), a classic over-instruction effect — extra clauses
  dilute the model's calibration and introduce new errors that outnumber the (zero) gains on
  the target cases.

## Conclusion — prompt disambiguation is NOT the lever
This confirms, decisively, the caveat attached to the wash diagnosis: **the two axes are
ambiguous because the underlying pragmatic judgment is genuinely hard for the model, not
because the prompt lacked the distinction.** Writing the distinction into the prompt (a) does
not give the model the capability to make it, and (b) adds complexity that hurts elsewhere.

This is now the **fourth prompt/topology intervention** to fail on the strong primary
(semantic_v1 ≈ 0, v3 ≈ 0, sequential v1/v2 ≤ 0, disambig −1 to −4). The consistent, repeated
result: **you cannot prompt-engineer past the strong-primary ceiling.** The only things that
moved it were a stronger *model* (tiny, +1, non-significant) and a weaker *primary* (large,
C3 +53).

## Recommendation
- **Revert to `semantic_v1`** for any real use — `semantic_v2_disambig` is strictly worse on
  the strong primary. Keep it in the codebase (opt-in, off by default) only as a documented
  negative.
- **Do not spend on a C3 disambig run** unless specifically wanted: the failure mode
  (relationship judgment is the hard problem + over-instruction noise) is primary-independent,
  so it is low expected value. If anything is run on C3 next, it should be the plain
  **G @ 4.1-mini** (stronger model on the larger recoverable slice), not disambig.
- **Accept the ceiling story as settled:** on the strong primary, neither prompts nor topology
  help; gains require a better primary (C3 regime) or a materially stronger model — and even
  the stronger model only nudges (non-significant) on the strong primary.

## Artifacts
- `experiment_G_ahmed_41mini_disambig/G_ahmed_41mini_disambig__*`
- Variant: `semantic_v2_disambig` in `src/prompts/{llm_lexical,polarity,contextual,intent}_prompt.py`
- Basis: `EXPERIMENT_G41_WASH_DIAGNOSIS.md` (the hypothesis this refutes),
  `EXPERIMENT_G_AHMED_GPT41MINI_RESULTS.md`.
