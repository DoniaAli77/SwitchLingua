# Design G on Ahmed at gpt-4.1-mini — Net Result

Full Design-G run on the Ahmed frozen primary with the agents upgraded from gpt-4o-mini to
**gpt-4.1-mini** (everything else identical: threshold 0.7, gate variant, semantic_v1,
w_primary=1.0, 84 escalated). Measures the **net** effect after the stronger model re-decides
all 84 escalated cases — not just the 18 it was diagnosed on. Date: 2026-07-02.

## Result

| system | accuracy | macro F1 | escalated acc | net vs primary |
|---|---|---|---|---|
| primary_only | 0.9254 | 0.9207 | 63/84 = 0.750 | — |
| G @ gpt-4o-mini | 0.9279 | 0.9257 | 65/84 = 0.774 | +2 |
| **G @ gpt-4.1-mini** | **0.9291** | 0.9248 | **66/84 = 0.786** | **+3** (W→C 10, C→W 7) |

- **0.9291 is the best full-set Ahmed number we've produced** (760/818) — but it is **+1
  sample over G@4o-mini** (759/818), not the +4 the failure-diagnostic hinted.
- Escalated 66/84 vs G@4o-mini's 65/84 → **+1 escalated.**
- Cost ≈ $0.13 (usage JSON shows $0.00 only because the cost table lacks 4.1-mini pricing;
  420 calls).

## Why the diagnostic's "4/18 fixed" became only +1 net
The targeted diagnostic showed 4.1-mini fixes 4 of the 18 failures. But a full run also
re-decides the **66 previously-correct** escalated cases, and the stronger model **broke some
of those**: it introduced new C→W errors (7 total vs 4o-mini's fewer) while adding the ~4
rescues. Net escalated moved 65 → 66: **+4 fixed − ~3 newly-broken ≈ +1.** This is the exact
caveat flagged in the diagnostic — a stronger model is not uniformly better; it re-rolls every
decision, so recall gains are partly offset by new mistakes.

## Significance — still not a real beat on the strong primary
McNemar vs primary_only (b=10, c=7): χ² ≈ 0.53, **p ≈ 0.47 — NOT significant.** Even with the
stronger model, Design G on the strong Ahmed primary remains **statistically indistinguishable
from doing nothing**, and **does not reach the 0.930 target** (0.9291). The model swap nudged
the point estimate up by 1 sample; it did not break the ceiling.

## The real takeaway
- **On the strong primary, a stronger model gives a genuine but tiny bump** (0.9279 → 0.9291,
  best-yet but non-significant). The recoverable slice the diagnostic found (compliance +
  obscured praise) is real but small, and partly cancelled by fresh errors.
- **The higher-value use of the stronger model is the WEAK C3 primary**, where the recoverable
  slice is much larger (231 escalated, primary only 0.54 there). G@4o-mini already got +53 on
  C3; a 4.1-mini upgrade has far more room to help there than on Ahmed's near-ceiling subset.
- Consistent with every prior result: the strong-primary ceiling is a hard wall; **model
  quality nudges it, topology doesn't, and the weak primary is where gains live.**

## Recommended next step
Run **Design G on C3 at gpt-4.1-mini** (~$0.45) — the one place the stronger model's
recoverable slice can actually compound. Compare vs G@4o-mini on C3 (0.7604 / esc 0.771).

## Artifacts
- `experiment_G_ahmed_gpt41mini/G_ahmed_41mini__{primary_only,full_pipeline}_*`
- Basis: `EXPERIMENT_G_STRONGER_MODEL_DIAGNOSTIC.md` (the 4/18 targeted diagnostic),
  `EXPERIMENT_G_C3_RESULTS.md`.
