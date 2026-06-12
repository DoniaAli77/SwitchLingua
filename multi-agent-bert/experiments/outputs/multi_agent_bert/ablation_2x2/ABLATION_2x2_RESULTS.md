# 2×2 Ablation Results — Fix #2 × Fix #3

XLM-R, full_agentic, threshold 0.8, EESA test (818), gpt-4o-mini. All four cells
**clean** (0 connection errors, 0 parse errors, 436 calls = 109×4 each). Total
spend ~$0.24. Date: 2026-06-12.

| Cell | w_primary | signal | acc | macro F1 | weighted F1 | pos / neg / neu F1 | esc | anchor % | cost |
|---|---|---|---|---|---|---|---|---|---|
| **A** original | 0 | OFF | 0.8460 | 0.8331 | 0.8445 | 0.907 / 0.808 / 0.784 | 109 | 62.4% | $0.0569 |
| **B** Fix #2 only | 1.0 | OFF | 0.8447 | 0.8316 | 0.8432 | 0.907 / 0.806 / 0.781 | 109 | 62.4% | $0.0574 |
| **C** Fix #3 only | 0 | ON | 0.8435 | 0.8312 | 0.8424 | 0.904 / 0.806 / 0.783 | 109 | 65.1% | $0.0640 |
| **D** Fix #2 + #3 | 1.0 | ON | 0.8423 | 0.8300 | 0.8413 | 0.903 / 0.804 / 0.783 | 109 | 66.1% | $0.0645 |

Reference: primary_only XLM-R = 0.8240 / 0.8088.

**anchor %** = among escalated samples, fraction where the final label equals the
primary_only prediction (higher ⇒ agents agree with / copy the primary more).

## Findings
1. **Accuracy: all four tied, within run-to-run GPT-4o-mini noise.** Full spread
   A→D is only −0.0037 acc (~3 of 818 samples). For full_agentic with *strong*
   real-LLM agents at threshold 0.8, neither fix moves accuracy meaningfully.
2. **Anchoring proxy confirms the Fix #3 risk.** Signal-ON cells (C, D) raise
   agent-vs-primary agreement **62.4% → 65.1% / 66.1%** (+3–4 pts): the
   primary-signal block **induces anchoring**, *and* it slightly lowers accuracy
   (C<A, D<B) while costing **+$0.007/run** in extra prompt tokens. The block buys
   more copying, not more accuracy, in this regime.
3. **Fix #2 is neutral here** (B≈A; anchoring unchanged at 62.4%). Combined with
   its **+0.064 acc paper_style win**, Fix #2's value is in the *weak-agent*
   regime and it is harmless with strong agents.
4. **Negative F1 protected** across all cells (0.808 → 0.804, flat within noise).

## Recommendation (data-driven)
- **Fix #2 (primary-aware consensus): keep default ON, `w_primary = 1.0`.** Big win
  for weak deterministic agents (paper_style), neutral for strong real-LLM agents.
- **Fix #3 (primary-signal block): keep default OFF.** On strong real-LLM
  sentiment agents it adds anchoring without accuracy benefit (and costs more). It
  *may* help **topic** (weaker 9-way agents, where adjudication has more to gain) —
  untested; revisit when topic data exists. Do not enable for strong-agent
  sentiment.

## Notes
- Conclusion is about full_agentic real-LLM sentiment at threshold 0.8. paper_style
  (weak agents) tells the opposite Fix-#2 story (big gain) — see
  `agent_prompt_audit/PRIMARY_AWARE_CONSENSUS_CHANGELOG.md`.
- Per-cell outputs under `ablation_2x2/{A,B,C,D}/`; run log `ablation_2x2.log`.
- Seam used: `--consensus_primary_weight {0|1.0}` + `--agents_use_primary_signal`
  (defaults unchanged: w_primary 1.0, signal off).
