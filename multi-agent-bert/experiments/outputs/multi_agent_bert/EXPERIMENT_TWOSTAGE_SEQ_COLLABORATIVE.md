# Sequential-Collaborative on a WEAK Primary (Two-Stage XLM-R + G2)

The parallel-vs-sequential comparisons so far used the strong Ahmed primary
(0.9254), where agents have little headroom. This repeats the collaborative test
on the **weak** two-stage XLM-R primary (GEN-960 → EESA, 0.8655) where parallel
agents demonstrably help — the case most favourable to sequential collaboration.
Same G2 selective gate, gpt-4.1-mini, threshold 0.90. Parallel G2 reused from
`experiment_twostage_g2_41mini`; collaborative is new. Date: 2026-07-19.

## Result

| mode | accuracy | macro F1 | escalated /138 |
|---|---|---|---|
| primary only | 0.8655 | 0.8579 | — |
| **parallel G2** | **0.8851** | **0.8784** | 105 |
| sequential-collaborative G2 | 0.8826 | 0.8751 | 103 |

8 predictions changed vs parallel (3 helped, 5 hurt, **net −2**), 0 fallbacks.

## Why this matters
This primary is weak, so parallel agents add a real **+0.0196 acc / +0.0205 F1**
over primary-only — genuine headroom. Even so, collaborative sequential chaining
is **slightly worse than parallel** (−0.0025 acc, −2 escalated samples). It does
not capture more of the available gain; it gives a little back. The changed
samples here skew toward pulling *correct negatives → neutral* (idx 164, 265,
567) — the opposite direction from the Ahmed over-calling, but the same net sign.

## Complete picture across primaries

| Primary | strength | parallel | seq-collaborative | Δ (esc) |
|---|---|---|---|---|
| Ahmed (Design C) | strong 0.925 | 63/84 | 61/84 | −2 |
| Ahmed (Design G) | strong 0.925 | 64/84 | 64/84 | 0 |
| Ahmed (G2) | strong 0.925 | 68/84 | 67/84 | −1 |
| Two-stage XLM-R (G2) | weak 0.866 | 105/138 | 103/138 | −2 |

**Collaborative sequential chaining is never better than parallel voting** — across
strong and weak primaries, small and large escalated sets. Voter independence is
protective regardless of primary strength.

## Reproduce
- `scripts/twostage_g2_seq_collaborative.py`; primary checkpoint
  `experiments/checkpoints/expTwoStage_gen960/fullEESA`; test
  `data/Sentiment/processed/eesa_sentiment_test.jsonl`.
- Flags: `sequential_chain=True`, `sequential_chain_style="collaborative"`.
- Artifacts: `experiment_twostage_g2_seq_collaborative/{collab_records.json,twostage_collab_report.txt}`.
