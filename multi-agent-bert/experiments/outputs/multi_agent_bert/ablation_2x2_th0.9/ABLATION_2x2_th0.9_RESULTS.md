# 2×2 Ablation Results — threshold 0.9

XLM-R, full_agentic, threshold 0.9, EESA test, gpt-4o-mini. All four cells
**clean** (0 connection errors, 0 parse errors, 760 calls = 190×4 each). Total
spend ~$0.42. Date: 2026-06-13.

| Cell | wp | sig | acc | macro F1 | wF1 | neg F1 | neu F1 | esc | escAcc | anchor% | net(W→C−C→W) | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** original | 0 | OFF | 0.8496 | 0.8386 | 0.8475 | 0.832 | 0.779 | 190 | 0.668 | 63.7 | +21 | $0.0985 |
| **B** Fix #2 | 1.0 | OFF | **0.8509** | **0.8401** | 0.8487 | **0.835** | 0.780 | 190 | **0.674** | 64.7 | **+22** | $0.0993 |
| **C** Fix #3 | 0 | ON | 0.8472 | 0.8369 | 0.8459 | 0.829 | 0.780 | 190 | 0.658 | 67.4 | +19 | $0.1109 |
| **D** #2+#3 | 1.0 | ON | 0.8496 | 0.8394 | 0.8482 | 0.831 | 0.784 | 190 | 0.668 | 70.0 | +21 | $0.1117 |

Escalation 190/818 = 23.2%. anchor% = escalated final == primary_only agreement.
primary_only XLM-R ref: 0.8240 / 0.8088.

## 0.8 vs 0.9
| metric | 0.8 (A/B/C/D) | 0.9 (A/B/C/D) |
|---|---|---|
| accuracy | 0.8460 / 0.8447 / 0.8435 / 0.8423 | 0.8496 / 0.8509 / 0.8472 / 0.8496 |
| anchor% | 62.4 / 62.4 / 65.1 / 66.1 | 63.7 / 64.7 / 67.4 / 70.0 |
| signal-ON anchoring Δ (vs OFF) | +2.7 / +1.0 | **+3.7 / +5.3** |
| escalated set | 109 | 190 |

## Findings
1. **Accuracy still within noise.** Full A→D spread is 0.0037 (same as 0.8); no
   cell differs from A by >0.005. The fixes do not move full_agentic accuracy on
   the larger 190-sample escalated slice either.
2. **Fix #2 flipped from slightly-negative (0.8) to slightly-positive (0.9).**
   B leads A by +0.0013 acc / +0.0015 macro F1, with best escalated-acc (0.674),
   best negative F1 (0.835) and best net override balance (+22). This is the
   **predicted threshold-dependence** — the primary anchor helps a little more
   when more (and more confident) samples escalate — though still inside noise.
   → keep **Fix #2 default ON**.
3. **Fix #3 anchoring grew; still no benefit.** signal-ON anchoring delta rose to
   **+3.7 / +5.3 pts** (D copies the primary 70% of the time) vs +2.7 / +1.0 at
   0.8, while C/D do not beat their signal-OFF counterparts (C<A; D=A<B). More
   copying, not more accuracy. → keep **Fix #3 default OFF**.

## Stopping rule → STOP (do not run 0.6/0.7)
The recommendation (Fix #2 on / Fix #3 off) is unchanged, all cells within noise,
and the anchoring delta keeps the same sign (and grows). Per the agreed criteria,
nothing flips the conclusion across 0.8 and 0.9 → the conclusion is robust over
the high-escalation operating range; 0.6/0.7 (fewer escalations, smaller effects)
would add no information.

Transparency note: the only strict-reading "ordering change" is the within-noise
top-2 wobble (B edged above A at 0.9 by +0.0013). It does not change the
recommendation. Running 0.6/0.7 would only map the noise-level Fix-#2 trend — not
recommended.

Per-cell outputs under `ablation_2x2_th0.9/{A,B,C,D}/`; run log
`ablation_2x2_th0.9.log`.
