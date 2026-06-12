# Multi-Agent BERT — Sentiment: Final Status

Date: 2026-06-13. Audit-driven fix program complete and validated for sentiment.

## Final decision (locked)
- **Fix #2 primary-aware consensus: ON, `w_primary = 1.0`** (current default).
- **Fix #3 primary-signal prompt block: OFF** (current default).
- **Router unchanged.** Threshold is a per-run knob, not changed in config.
- No further paid ablations now; do **not** run 0.6/0.7.

These match the shipped defaults — no code change required to honour the decision.

## Best performance setting
**XLM-R + full_agentic + GPT-4o-mini + threshold 0.9 + w_primary=1.0 +
primary_signal OFF** → **0.8509 accuracy / 0.8401 macro F1** (weighted F1 0.8487,
negative F1 0.835), EESA test. (= cell B of the 0.9 2×2.)

## Practical lower-cost setting
**Threshold 0.8 remains acceptable** (fewer escalations → ~$0.057 vs ~$0.099 per
run; ~109 vs ~190 LLM-handled samples) at 0.8447 acc / 0.8316 macro F1.
**0.9 is the strongest result.**

## Reference points (EESA test, XLM-R)
| Setting | acc | macro F1 |
|---|---|---|
| primary_only (no agents) | 0.8240 | 0.8088 |
| full_agentic @0.8, Fix #2 on, signal off | 0.8447 | 0.8316 |
| **full_agentic @0.9, Fix #2 on, signal off (BEST)** | **0.8509** | **0.8401** |
| paper_style @0.9 (Fix #2 win vs agents-only) | 0.8056 | 0.7882 |

## What's implemented & validated
- Generic (task-config-driven) prompts; no sentiment/topic labels hardcoded.
- No-vote / abstain fallback (no `labels[0]` bias).
- Primary-aware consensus (Fix #2) with non-positional tie-break; config-gated
  `w_primary` (default 1.0; seam: `--consensus_primary_weight`).
- Optional primary-signal block (Fix #3), config-gated, **default off** (seam:
  `--agents_use_primary_signal`).
- Test suite: **897 passing**, fully offline. Two clean 2×2 ablations (0.8, 0.9)
  confirm the decision; anchoring proxy shows Fix #3 induces copying without gain
  on strong agents.

## Remaining future work (not for the current sentiment phase)
- **Router / threshold** — per-task threshold + margin/entropy signal, mainly for
  the **topic** phase (9-class escalation differs from 3-class).
- **Confidence calibration (M3)** — agent self-confidences are uncalibrated;
  address only if it proves to matter.
- **Revisit the primary-signal block (Fix #3) for topic** — weaker 9-way agents
  may benefit from adjudication; re-run the C/D cells when topic data exists.

Detailed evidence: `agent_prompt_audit/` (audit + per-fix changelogs/proposals),
`ablation_2x2/` and `ablation_2x2_th0.9/` (ablation results).
